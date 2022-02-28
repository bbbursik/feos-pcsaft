use super::association::N0_CUTOFF;
use super::polar::{pair_integral_ij, triplet_integral_ijk};
use crate::eos::association::{assoc_site_frac_a, assoc_site_frac_ab};
use crate::eos::dispersion::{A0, A1, A2, B0, B1, B2};
use crate::eos::polar::{AD, AQ, BD, BQ, CD, CQ, PI_SQ_43};
use crate::parameters::PcSaftParameters;
use feos_core::{EosError, EosResult};
use feos_dft::fundamental_measure_theory::{FMTProperties, FMTVersion};
use feos_dft::{
    FunctionalContributionDual, WeightFunction, WeightFunctionInfo, WeightFunctionShape,
};
use ndarray::*;
use num_dual::*;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::rc::Rc;

use feos_dft::entropy_scaling::EntropyScalingFunctionalContribution;

const PI36M1: f64 = 1.0 / (36.0 * PI);
const N3_CUTOFF: f64 = 1e-5;

#[derive(Clone)]
pub struct PureFMTAssocFunctional {
    parameters: Rc<PcSaftParameters>,
    version: FMTVersion,
}

impl PureFMTAssocFunctional {
    pub fn new(parameters: Rc<PcSaftParameters>, version: FMTVersion) -> Self {
        Self {
            parameters,
            version,
        }
    }
}

impl<N: DualNum<f64> + ScalarOperand> FunctionalContributionDual<N> for PureFMTAssocFunctional {
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let r = self.parameters.hs_diameter(temperature) * 0.5;
        WeightFunctionInfo::new(arr1(&[0]), false).extend(
            vec![
                WeightFunctionShape::Delta,
                WeightFunctionShape::Theta,
                WeightFunctionShape::DeltaVec,
            ]
            .into_iter()
            .map(|s| WeightFunction {
                prefactor: self.parameters.m.mapv(|m| m.into()),
                kernel_radius: r.clone(),
                shape: s,
            })
            .collect(),
            false,
        )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>> {
        let p = &self.parameters;

        // weighted densities
        let n2 = weighted_densities.index_axis(Axis(0), 0);
        let n3 = weighted_densities.index_axis(Axis(0), 1);
        let n2v = weighted_densities.slice_axis(Axis(0), Slice::new(2, None, 1));

        // temperature dependent segment radius
        let r = self.parameters.hs_diameter(temperature)[0] * 0.5;

        // auxiliary variables
        if n3.iter().any(|n3| n3.re() > 1.0) {
            return Err(EosError::IterationFailed(String::from(
                "PureFMTAssocFunctional",
            )));
        }
        let ln31 = n3.mapv(|n3| (-n3).ln_1p());
        let n3rec = n3.mapv(|n3| n3.recip());
        let n3m1 = n3.mapv(|n3| -n3 + 1.0);
        let n3m1rec = n3m1.mapv(|n3m1| n3m1.recip());
        let n1 = n2.mapv(|n2| n2 / (r * 4.0 * PI));
        let n0 = n2.mapv(|n2| n2 / (r * r * 4.0 * PI));
        let n1v = n2v.mapv(|n2v| n2v / (r * 4.0 * PI));

        let (n1n2, n2n2) = match self.version {
            FMTVersion::WhiteBear => (
                &n1 * &n2 - (&n1v * &n2v).sum_axis(Axis(0)),
                &n2 * &n2 - (&n2v * &n2v).sum_axis(Axis(0)) * 3.0,
            ),
            FMTVersion::AntiSymWhiteBear => {
                let mut xi2 = (&n2v * &n2v).sum_axis(Axis(0)) / n2.map(|n| n.powi(2));
                xi2.iter_mut().for_each(|x| {
                    if x.re() > 1.0 {
                        *x = N::one()
                    }
                });
                (
                    &n1 * &n2 - (&n1v * &n2v).sum_axis(Axis(0)),
                    &n2 * &n2 * xi2.mapv(|x| (-x + 1.0).powi(3)),
                )
            }
            _ => unreachable!(),
        };

        // The f3 term contains a 0/0, therefore a taylor expansion is used for small values of n3
        let mut f3 = (&n3m1 * &n3m1 * &ln31 + n3) * &n3rec * n3rec * &n3m1rec * &n3m1rec;
        f3.iter_mut().zip(n3).for_each(|(f3, &n3)| {
            if n3.re() < N3_CUTOFF {
                *f3 = (((n3 * 35.0 / 6.0 + 4.8) * n3 + 3.75) * n3 + 8.0 / 3.0) * n3 + 1.5;
            }
        });
        let mut phi = -(&n0 * &ln31) + n1n2 * &n3m1rec + n2n2 * n2 * PI36M1 * f3;

        // association
        if p.nassoc == 1 {
            let mut xi = -(&n2v * &n2v).sum_axis(Axis(0)) / (&n2 * &n2) + 1.0;
            xi.iter_mut().zip(&n2).for_each(|(xi, &n2)| {
                if n2.re() < N0_CUTOFF * 4.0 * PI * p.m[0] * r.re().powi(2) {
                    *xi = N::one();
                }
            });

            let k = &n2 * &n3m1rec * r;
            let deltarho = (((&k / 18.0 + 0.5) * &k * &xi + 1.0) * n3m1rec)
                * ((temperature.recip() * p.epsilon_k_aibj[(0, 0)]).exp_m1()
                    * (p.sigma[0].powi(3) * p.kappa_aibj[(0, 0)]))
                * (&n0 / p.m[0] * &xi);

            let f = |x: N| x.ln() - x * 0.5 + 0.5;
            phi = phi
                + if p.nb[0] > 0.0 {
                    let xa = deltarho.mapv(|d| assoc_site_frac_ab(d, p.na[0], p.nb[0]));
                    let xb = (xa.clone() - 1.0) * p.na[0] / p.nb[0] + 1.0;
                    (n0 / p.m[0] * xi) * (xa.mapv(f) * p.na[0] + xb.mapv(f) * p.nb[0])
                } else {
                    let xa = deltarho.mapv(|d| assoc_site_frac_a(d, p.na[0]));
                    n0 / p.m[0] * xi * (xa.mapv(f) * p.na[0])
                };
        }

        Ok(phi)
    }
}

impl fmt::Display for PureFMTAssocFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pure FMT+association")
    }
}

#[derive(Clone)]
pub struct PureChainFunctional {
    parameters: Rc<PcSaftParameters>,
}

impl PureChainFunctional {
    pub fn new(parameters: Rc<PcSaftParameters>) -> Self {
        Self { parameters }
    }
}

impl<N: DualNum<f64> + ScalarOperand> FunctionalContributionDual<N> for PureChainFunctional {
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        WeightFunctionInfo::new(arr1(&[0]), true)
            .add(
                WeightFunction::new_scaled(d.clone(), WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction {
                    prefactor: (&self.parameters.m / 8.0).mapv(|x| x.into()),
                    kernel_radius: d.clone(),
                    shape: WeightFunctionShape::Theta,
                },
                false,
            )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        _: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>> {
        let rho = weighted_densities.index_axis(Axis(0), 0);
        // negative lambdas lead to nan, therefore the absolute value is used
        let lambda = weighted_densities
            .index_axis(Axis(0), 1)
            .map(|&l| if l.re() < 0.0 { -l } else { l } + N::from(f64::EPSILON));
        let eta = weighted_densities.index_axis(Axis(0), 2);

        let y = eta.mapv(|eta| (eta * 0.5 - 1.0) / (eta - 1.0).powi(3));
        Ok(-(y * lambda).mapv(|x| (x.ln() - 1.0) * (self.parameters.m[0] - 1.0)) * rho)
    }
}

impl fmt::Display for PureChainFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pure chain")
    }
}

#[derive(Clone)]
pub struct PureAttFunctional {
    parameters: Rc<PcSaftParameters>,
}

impl PureAttFunctional {
    pub fn new(parameters: Rc<PcSaftParameters>) -> Self {
        Self { parameters }
    }
}

impl<N: DualNum<f64> + ScalarOperand> FunctionalContributionDual<N> for PureAttFunctional {
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        const PSI: f64 = 1.3862; // Homosegmented DFT (Sauer2017)
        WeightFunctionInfo::new(arr1(&[0]), false).add(
            WeightFunction::new_scaled(d * PSI, WeightFunctionShape::Theta),
            false,
        )
    }

    fn weight_functions_pdgt(&self, temperature: N) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        const PSI: f64 = 1.3286; // pDGT (Rehner2018)
        WeightFunctionInfo::new(arr1(&[0]), false).add(
            WeightFunction::new_scaled(d * PSI, WeightFunctionShape::Theta),
            false,
        )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>> {
        let p = &self.parameters;
        let rho = weighted_densities.index_axis(Axis(0), 0);

        // temperature dependent segment radius
        let d = p.hs_diameter(temperature)[0];

        let eta = rho.mapv(|rho| rho * FRAC_PI_6 * p.m[0] * d.powi(3));
        let m1 = (p.m[0] - 1.0) / p.m[0];
        let m2 = m1 * (p.m[0] - 2.0) / p.m[0];
        let e = temperature.recip() * p.epsilon_k[0];
        let s3 = p.sigma[0].powi(3);

        // I1, I2 and C1
        let mut i1: Array1<N> = Array::zeros(eta.raw_dim());
        let mut i2: Array1<N> = Array::zeros(eta.raw_dim());
        for i in 0..=6 {
            i1 = i1 + eta.mapv(|eta| eta.powi(i as i32) * (A0[i] + m1 * A1[i] + m2 * A2[i]));
            i2 = i2 + eta.mapv(|eta| eta.powi(i as i32) * (B0[i] + m1 * B1[i] + m2 * B2[i]));
        }
        let c1 = eta.mapv(|eta| {
            ((eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4) * p.m[0]
                + (eta * 20.0 - eta.powi(2) * 27.0 + eta.powi(3) * 12.0 - eta.powi(4) * 2.0)
                    / ((eta - 1.0) * (eta - 2.0)).powi(2)
                    * (1.0 - p.m[0])
                + 1.0)
                .recip()
        });
        let mut phi = rho.mapv(|rho| -(rho * p.m[0]).powi(2) * e * s3 * PI)
            * (i1 * 2.0 + c1 * i2.mapv(|i2| i2 * p.m[0] * e));

        // dipoles
        if p.ndipole > 0 {
            let mu2_term = e * s3 * p.mu2[0];
            let m = p.m[0].min(2.0);
            let m1 = (m - 1.0) / m;
            let m2 = m1 * (m - 2.0) / m;

            let phi2 = -(&rho * &rho)
                * pair_integral_ij(m1, m2, &eta, &AD, &BD, e)
                * (mu2_term * mu2_term / s3 * PI);
            let phi3 = -(&rho * &rho * rho)
                * triplet_integral_ijk(m1, m2, &eta, &CD)
                * (mu2_term * mu2_term * mu2_term / s3 * PI_SQ_43);

            let mut phi_d = &phi2 * &phi2 / (&phi2 - &phi3);
            phi_d.iter_mut().zip(phi2.iter()).for_each(|(p, &p2)| {
                if p.re().is_nan() {
                    *p = p2;
                }
            });
            phi += &phi_d;
        }

        // quadrupoles
        if p.nquadpole > 0 {
            let q2_term = e * p.sigma[0].powi(5) * p.q2[0];
            let m = p.m[0].min(2.0);
            let m1 = (m - 1.0) / m;
            let m2 = m1 * (m - 2.0) / m;

            let phi2 = -(&rho * &rho)
                * pair_integral_ij(m1, m2, &eta, &AQ, &BQ, e)
                * (q2_term * q2_term / p.sigma[0].powi(7) * PI * 0.5625);
            let phi3 = (&rho * &rho * rho)
                * triplet_integral_ijk(m1, m2, &eta, &CQ)
                * (q2_term * q2_term * q2_term / s3.powi(3) * PI * PI * 0.5625);

            let mut phi_q = &phi2 * &phi2 / (&phi2 - &phi3);
            phi_q.iter_mut().zip(phi2.iter()).for_each(|(p, &p2)| {
                if p.re().is_nan() {
                    *p = p2;
                }
            });
            phi += &phi_q;
        }

        Ok(phi)
    }
}

impl fmt::Display for PureAttFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pure attractive")
    }
}

impl EntropyScalingFunctionalContribution for PureFMTAssocFunctional {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64> {
        let r = self.parameters.hs_diameter(temperature) * 0.5;
        WeightFunctionInfo::new(self.parameters.component_index().clone(), false)
            .add(
                WeightFunction::new_scaled(r.clone()*0.5, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*0.6, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*0.7, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*0.8, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*0.9, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.0, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.1, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.2, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.3, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.4, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.5, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.6, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.7, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.8, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.9, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*2.0, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*2.5, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*3.0, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*4.0, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*5.0, WeightFunctionShape::Theta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*0.5, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*0.6, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*0.7, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*0.8, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*0.9, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.0, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.1, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.2, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.3, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.4, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.5, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.6, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.7, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.8, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*1.9, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*2.0, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*2.5, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*3.0, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*4.0, WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction::new_scaled(r.clone()*5.0, WeightFunctionShape::Delta),
                false,
            )
            













            // .add(
            //     WeightFunction::new_scaled(r.clone(), WeightFunctionShape::Delta),
            //     false,
            // )
            
        // .add(
        //     WeightFunction::new_scaled(r.clone(), WeightFunctionShape::DeltaVec),
        //     false,
        // )
    }

    // fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64> {
    //     let r = self.parameters.hs_diameter(temperature) * 0.5;
    //     let psis = Array::linspace(0.5, 2.5, 21);
    //     let mut wf_info = WeightFunctionInfo::new(self.parameters.component_index(), false);
    //     // for psi in psis{
    //     //     let wf_info = wf_info.add(
    //     //         WeightFunction::new_scaled(&r*psi, WeightFunctionShape::Theta),
    //     //         true,
    //     //     );
    //     psis.iter().for_each(|&psi| {
    //         wf_info.add(
    //                 WeightFunction::new_scaled(r.clone()* psi, WeightFunctionShape::Theta),
    //                 true,
    //             );
    //         }
    //     );

    
        
    //     println!("psis are used");
    
    //     // for psi in psis{
    //     //     let wf_info = wf_info.add(
    //     //         WeightFunction::new_scaled(&r*psi, WeightFunctionShape::Delta),
    //     //         true,
    //     //     );
    //     // }
        
    //     wf_info
        
    // }

}

impl EntropyScalingFunctionalContribution for PureChainFunctional {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64> {
        let d = self.parameters.hs_diameter(temperature);
        WeightFunctionInfo::new(self.parameters.component_index().clone(), false).add(
            WeightFunction::new_scaled(d.clone(), WeightFunctionShape::Theta),
            true,
        )
    }
}

impl EntropyScalingFunctionalContribution for PureAttFunctional {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64> {
        let d = self.parameters.hs_diameter(temperature);
        const PSI: f64 = 1.3862; // Homosegmented DFT (Sauer2017)
        WeightFunctionInfo::new(self.parameters.component_index().clone(), false).add(
            WeightFunction::new_scaled(d * PSI, WeightFunctionShape::Theta),
            true,
        )
    }
}
