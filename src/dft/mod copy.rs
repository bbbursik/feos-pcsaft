use crate::eos::omega22;
use crate::eos::PcSaftOptions;
use crate::parameters::PcSaftParameters;
use association::AssociationFunctional;
use dispersion::AttractiveFunctional;
use feos_core::parameter::Parameter;
use feos_core::EosError;
use feos_core::EosUnit;
use feos_core::MolarWeight;
use feos_dft::adsorption::FluidParameters;
use feos_dft::entropy_scaling::EntropyScalingFunctional;
use feos_dft::entropy_scaling::EntropyScalingFunctionalContribution;
use feos_dft::fundamental_measure_theory::{FMTContribution, FMTProperties, FMTVersion};
use feos_dft::solvation::PairPotential;
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctional, DFT};
use hard_chain::ChainFunctional;
use ndarray::Ix1;
use ndarray::Ix2;
use ndarray::Zip;
use ndarray::{Array, Array1, Array2, Axis as Axis_nd, RemoveAxis};
use num_dual::DualNum;
use num_traits::One;
use pure_saft_functional::*;
use quantity::si::*;
use std::f64::consts::{FRAC_PI_6, PI};
use std::rc::Rc;

mod association;
mod dispersion;
mod hard_chain;
mod polar;
mod pure_saft_functional;

pub struct PcSaftFunctional {
    pub parameters: Rc<PcSaftParameters>,
    fmt_version: FMTVersion,
    options: PcSaftOptions,
    contributions: Vec<Box<dyn FunctionalContribution>>,
    entropy_scaling_contributions: Vec<Box<dyn EntropyScalingFunctionalContribution>>,
}

impl PcSaftFunctional {
    pub fn new(parameters: Rc<PcSaftParameters>) -> DFT<Self> {
        Self::new_with_options(parameters, FMTVersion::WhiteBear, PcSaftOptions::default())
    }

    pub fn new_full(parameters: Rc<PcSaftParameters>, fmt_version: FMTVersion) -> DFT<Self> {
        Self::new_with_options(parameters, fmt_version, PcSaftOptions::default())
    }

    fn new_with_options(
        parameters: Rc<PcSaftParameters>,
        fmt_version: FMTVersion,
        saft_options: PcSaftOptions,
    ) -> DFT<Self> {
        let mut contributions: Vec<Box<dyn FunctionalContribution>> = Vec::with_capacity(4);
        let mut entropy_scaling_contributions: Vec<Box<dyn EntropyScalingFunctionalContribution>> =
            Vec::with_capacity(4);

        if matches!(
            fmt_version,
            FMTVersion::WhiteBear | FMTVersion::AntiSymWhiteBear
        ) && parameters.m.len() == 1
        {
            let fmt_assoc = PureFMTAssocFunctional::new(parameters.clone(), fmt_version);
            contributions.push(Box::new(fmt_assoc.clone()));
            entropy_scaling_contributions.push(Box::new(fmt_assoc.clone()));
            if parameters.m.iter().any(|&mi| mi > 1.0) {
                let chain = PureChainFunctional::new(parameters.clone());
                contributions.push(Box::new(chain.clone()));
                entropy_scaling_contributions.push(Box::new(chain.clone()));
            }
            let att = PureAttFunctional::new(parameters.clone());
            contributions.push(Box::new(att.clone()));
            entropy_scaling_contributions.push(Box::new(att.clone()));
        } else {
            // Hard sphere contribution
            let hs = FMTContribution::new(&parameters, fmt_version);
            contributions.push(Box::new(hs.clone()));
            entropy_scaling_contributions.push(Box::new(hs.clone()));

            // Hard chains
            if parameters.m.iter().any(|&mi| !mi.is_one()) {
                let chain = ChainFunctional::new(parameters.clone());
                contributions.push(Box::new(chain.clone()));
                entropy_scaling_contributions.push(Box::new(chain.clone()));
            }

            // Dispersion
            let att = AttractiveFunctional::new(parameters.clone());
            contributions.push(Box::new(att.clone()));
            entropy_scaling_contributions.push(Box::new(att.clone()));

            // Association
            if parameters.nassoc > 0 {
                let assoc = AssociationFunctional::new(
                    parameters.clone(),
                    saft_options.max_iter_cross_assoc,
                    saft_options.tol_cross_assoc,
                );
                contributions.push(Box::new(assoc.clone()));
                entropy_scaling_contributions.push(Box::new(assoc.clone()));
            }
        }

        let func = Self {
            parameters: parameters.clone(),
            fmt_version,
            options: saft_options,
            contributions,
            entropy_scaling_contributions,
        };

        DFT::new_homosegmented(func, &parameters.m)
    }
}

impl HelmholtzEnergyFunctional for PcSaftFunctional {
    fn subset(&self, component_list: &[usize]) -> DFT<Self> {
        Self::new_with_options(
            Rc::new(self.parameters.subset(component_list)),
            self.fmt_version,
            self.options,
        )
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.parameters.m * self.parameters.sigma.mapv(|v| v.powi(3)) * moles)
                .sum()
    }

    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        &self.contributions
    }
}

impl MolarWeight<SIUnit> for PcSaftFunctional {
    fn molar_weight(&self) -> SIArray1 {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

impl FMTProperties for PcSaftParameters {
    fn component_index(&self) -> Array1<usize> {
        Array::from_shape_fn(self.m.len(), |i| i)
    }

    fn chain_length(&self) -> Array1<f64> {
        self.m.clone()
    }

    fn hs_diameter<D: DualNum<f64>>(&self, temperature: D) -> Array1<D> {
        self.hs_diameter(temperature)
    }
}

impl FluidParameters for PcSaftFunctional {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        self.parameters.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.parameters.sigma
    }

    fn m(&self) -> Array1<f64> {
        self.parameters.m.clone()
    }
}

impl PairPotential for PcSaftFunctional {
    fn pair_potential(&self, r: &Array1<f64>) -> Array2<f64> {
        let sigma = &self.parameters.sigma;
        Array::from_shape_fn((self.parameters.m.len(), r.len()), |(i, j)| {
            4.0 * self.parameters.epsilon_k[i]
                * ((sigma[i] / r[j]).powi(12) - (sigma[i] / r[j]).powi(6))
        })
    }
}

impl EntropyScalingFunctional<SIUnit> for PcSaftFunctional {
    fn entropy_scaling_contributions(&self) -> &[Box<dyn EntropyScalingFunctionalContribution>] {
        &self.entropy_scaling_contributions
    }

    fn viscosity_reference<D>(
        &self,
        density: &SIArray<D::Larger>,
        temperature: SINumber,
    ) -> Result<SIArray<Ix1>, EosError> {
        // Extracting parameters and molar weight
        let p = &self.parameters;
        let mw = &p.molarweight;
        let n_comp = mw.len();

        // Pure references for each component (do only depend on temperature);
        // one reference per component, no grid distribution required
        let ce_eos: Array1<f64> = (0..n_comp)
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN)
                    .into_value()
                    .unwrap();
                (5.0 / 16.0
                    * (mw[i] * GRAM / MOL * KB / NAV * temperature / PI)
                        .sqrt()
                        .unwrap()
                    / omega22(tr)
                    / (p.sigma[i] * ANGSTROM).powi(2))
                .to_reduced(SIUnit::reference_viscosity())
                .unwrap()
            })
            .collect();

        // Factor `phi_ij`, no grid distribution required
        let mut phi = Array2::zeros((n_comp, n_comp));
        for ((i, j), phi_ij) in phi.indexed_iter_mut() {
            *phi_ij = (1.0 + (ce_eos[i] / ce_eos[j]).sqrt() * (mw[j] / mw[i]).powf(1.0 / 4.0))
                .powi(2)
                / (8.0 * (1.0 + mw[i] / mw[j])).sqrt();
        }

        // Mole fraction at every grid point
        let x = (density / &density.sum_axis(Axis_nd(0))).into_value()?; //.into_dimensionality().unwrap();

        //
        let visc_ref = Zip::from(x.lanes(Axis_nd(0))).map_collect(|x| {
            // Sum over `j` at every grid point
            let phi_i = phi
                .outer_iter()
                .map(|v| (&v * &x).sum())
                .collect::<Array1<f64>>();
            (&x * &ce_eos / &phi_i).sum()
        });

        // Return
        Ok(visc_ref * SIUnit::reference_viscosity())
    }

    fn viscosity_correlation<D>(
        &self,
        s_res: &Array<f64, Ix1>,
        density: &SIArray<Ix2>,
    ) -> Result<Array<f64, Ix1>, EosError> {
        // Extract references to viscosity parameters
        let coefficients = self
            .parameters
            .viscosity
            .as_ref()
            .expect("Missing viscosity coefficients");

        // Mole fraction at every grid point
        let x = (density / &density.sum_axis(Axis_nd(0))).into_value()?;

        // Scale residual entropy with mean chain length
        let mut m = Array::zeros(density.raw_dim());
        for mut lane in m.lanes_mut(Axis_nd(0)) {
            lane.assign(&self.parameters.m);
        }
        let m = (&x * &m).sum_axis(Axis_nd(0));
        let s = s_res / &m;

        // Mixture parameters
        let mut pref = Array::zeros(x.raw_dim());
        for mut lane in pref.lanes_mut(Axis_nd(0)) {
            lane.assign(&self.parameters.m);
        }
        Zip::from(pref.lanes_mut(Axis_nd(0)))
            .and(x.lanes(Axis_nd(0)))
            .for_each(|mut pl, xl| pl *= &xl);
        pref = &pref / &m;

        let a = Zip::from(x.lanes(Axis_nd(0))).map_collect(|xl| (&coefficients.row(0) * &xl).sum());
        let b =
            Zip::from(pref.lanes(Axis_nd(0))).map_collect(|pl| (&coefficients.row(1) * &pl).sum());
        let c =
            Zip::from(pref.lanes(Axis_nd(0))).map_collect(|pl| (&coefficients.row(2) * &pl).sum());
        let d =
            Zip::from(pref.lanes(Axis_nd(0))).map_collect(|pl| (&coefficients.row(3) * &pl).sum());

        // Return
        Ok(((d * &s + c) * &s + b) * s + a)
        // Ok(a + b * &s + c * &s.powi(2) + d * &s.powi(3))
    }
}
