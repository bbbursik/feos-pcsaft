use crate::eos::PcSaftOptions;
use crate::parameters::PcSaftParameters;
use association::AssociationFunctional;
use dispersion::AttractiveFunctional;
use feos_core::MolarWeight;
use feos_dft::adsorption::FluidParameters;
use feos_dft::fundamental_measure_theory::{FMTContribution, FMTProperties, FMTVersion};
use feos_dft::solvation::PairPotential;
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctional, DFT};
use hard_chain::ChainFunctional;
use ndarray::{Array, Array1, Array2};
use num_dual::DualNum;
use num_traits::One;
use pure_saft_functional::*;
use quantity::si::*;
use std::f64::consts::FRAC_PI_6;
use std::rc::Rc;

mod association;
mod dispersion;
mod hard_chain;
mod polar;
mod pure_saft_functional;

#[derive(Copy, Clone)]
enum PcSaftFunctionalVariants {
    Pure,
    FullVec,
    FullScal,
}

impl PcSaftFunctionalVariants {
    fn from_components(n: usize) -> Self {
        if n > 1 {
            Self::FullVec
        } else {
            Self::Pure
        }
    }

    fn new_full(use_vectors: bool) -> Self {
        if use_vectors {
            Self::FullVec
        } else {
            Self::FullScal
        }
    }
}

pub struct PcSaftFunctional {
    pub parameters: Rc<PcSaftParameters>,
    variant: PcSaftFunctionalVariants,
    options: PcSaftOptions,
    contributions: Vec<Box<dyn FunctionalContribution>>,
}

impl PcSaftFunctional {
    pub fn new(parameters: PcSaftParameters) -> DFT<Self> {
        let n: usize = parameters.m.len();
        Self::new_with_options(
            parameters,
            PcSaftFunctionalVariants::from_components(n),
            PcSaftOptions::default(),
        )
    }

    pub fn new_full(parameters: PcSaftParameters, use_vectors: bool) -> DFT<Self> {
        Self::new_with_options(
            parameters,
            PcSaftFunctionalVariants::new_full(use_vectors),
            PcSaftOptions::default(),
        )
    }

    fn new_with_options(
        parameters: PcSaftParameters,
        variant: PcSaftFunctionalVariants,
        saft_options: PcSaftOptions,
    ) -> DFT<Self> {
        DFT::new_homosegmented(
            Self::with_options(parameters.clone(), variant, saft_options),
            &parameters.m,
        )
    }

    fn with_options(
        parameters: PcSaftParameters,
        variant: PcSaftFunctionalVariants,
        saft_options: PcSaftOptions,
    ) -> Self {
        let parameters = Rc::new(parameters);

        let mut contributions: Vec<Box<dyn FunctionalContribution>> = Vec::with_capacity(4);

        if let PcSaftFunctionalVariants::Pure = variant {
            let fmt_assoc = PureFMTAssocFunctional::new(parameters.clone());
            contributions.push(Box::new(fmt_assoc.clone()));
            if parameters.m.iter().any(|&mi| mi > 1.0) {
                let chain = PureChainFunctional::new(parameters.clone());
                contributions.push(Box::new(chain.clone()));
            }
            let att = PureAttFunctional::new(parameters.clone());
            contributions.push(Box::new(att.clone()));
        } else {
            // Hard sphere contribution
            let fmt_version = match variant {
                PcSaftFunctionalVariants::FullVec => FMTVersion::WhiteBear,
                _ => FMTVersion::KierlikRosinberg,
            };
            let hs = FMTContribution::new(&parameters, fmt_version);
            contributions.push(Box::new(hs.clone()));

            // Hard chains
            if parameters.m.iter().any(|&mi| !mi.is_one()) {
                let chain = ChainFunctional::new(parameters.clone());
                contributions.push(Box::new(chain.clone()));
            }

            // Dispersion
            let att = AttractiveFunctional::new(parameters.clone());
            contributions.push(Box::new(att.clone()));

            // Association
            if parameters.nassoc > 0 {
                let assoc = AssociationFunctional::new(
                    parameters.clone(),
                    saft_options.max_iter_cross_assoc,
                    saft_options.tol_cross_assoc,
                );
                contributions.push(Box::new(assoc.clone()));
            }
        }

        Self {
            parameters,
            variant,
            options: saft_options,
            contributions,
        }
    }
}

impl HelmholtzEnergyFunctional for PcSaftFunctional {
    fn subset(&self, component_list: &[usize]) -> DFT<Self> {
        Self::new_with_options(
            self.parameters.subset(component_list),
            self.variant,
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
