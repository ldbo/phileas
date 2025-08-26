---
title: '*Phileas* — reproducible automation for hardware security experiments'
tags:
  - Python
  - fault injection
  - side-channel analysis
  - reproducibility
  - transparency
  - adaptability
  - test and measurements
date: 26 August 2025
authors:
  - name: Louis Dubois
    affiliation: '1, 2, 3'

affiliations:
  - index: 1
    name: Agence nationale de la sécurité des systèmes d'information, France
  - index: 2
    name: Direction Générale de l'Armement, France
    ror: 04wsqd844
  - index: 3
    name: Laboratory of Computer Science, Robotics and Microelectronics of Montpellier (LIRMM)
    ror: 013yean28

bibliography: paper.bib
---


# Summary

Hardware security is a cybersecurity field that relies on physical measurements of operating electronic devices to extract sensitive information. These measurements include, among others, electromagnetic field magnitude and integrated circuit current consumption. Most acquisitions involve large datasets — which can reach hundreds of gigabytes and above — with acquisition campaigns lasting from days to weeks[@lisovetsLetsTakeIt2021]. *Phileas* is an open-source Python package that automates this acquisition process, allowing security researchers and hardware designers to focus on data analysis. This provides the scientific community with a tool that improves the reproducibility, extensibility, and traceability of hardware security experiments.

First, it offers standardization of the representation of an acquisition campaign. This allows to easily share work with other researchers, guaranteeing the traceability of the datasets. A campaign is represented by a simple YAML configuration file that specifies the sequence of configurations applied to each of the instruments — that can be typical test and measurement (T&M) devices, the target system, or any other configurable item — on the bench. *Phileas* guarantees that, for a given configuration file, this sequence will be the same, even with complex parameters search methods such as randomization and shuffling. This ensures another researcher can reproduce the experiment on their own bench with minimal changes. This architecture also eases the extension of prior datasets to new setups, or to new experimental parameters. Finally, *Phileas* leverages the standardized representation of an experiment to guarantee that the resulting datasets are properly formatted and annotated.

All in all, *Phileas* aims at making acquisition scripts more reliable, and datasets easily shareable with other researchers in the community. This is a major - and, up to now, missing - software building block toward the reproducibility of hardware security research.

# Statement of need

Hardware security experiments generate massive datasets through weeks-long acquisition campaigns. Yet, the field lacks standardized tools for reliable and reproducible data collection. This hinders the reproducibility of the research process: researchers cannot reliably replicate experiments, extend existing datasets, or share methodologies effectively.

The field is at the intersection of computer science, electronic engineering, and cryptography. It studies the impacts of the physical implementations of algorithms on their security. Any software system manipulating sensitive data is eventually run by processors, microcontrollers or FPGAs, which are, in the end, physical systems executing computations. As a consequence, by measuring or modifying the state of these physical systems, it is possible to infer or change the data they handle. This can be achieved using several methods, including oscilloscopes and probes to measure their consumption [@mangardPowerAnalysisAttacks2007], spectrum analyzers and near-field antennas to measure the electromagnetic field they radiate [@gandolfiElectromagneticAnalysisConcrete2001], or focused lasers to induce photo-currents and modify their behavior [@skorobogatovOpticalFaultInduction2003]. Hardware cybersecurity research concentrates on two main attack vectors: fault injection attacks — active perturbation of systems to induce computational errors — and side channel attacks — passive extraction of secrets through physical measurements.

A typical hardware security experiment involves the following steps:

  1. A target algorithm, or software system, is selected, and implemented on a device;
  2. An experiment bench is set up, allowing to configure the target and perform stimuli and measurements on the device, as shown in \autoref{fig:bench};
  3. The experiment is automated through custom scripts that configure instruments and target (often across hundreds of thousands of parameter combinations), acquire measurements and store the resulting data;
  4. This dataset is analyzed using statistical methods (*e.g.* simple and differential power analysis [@kocherIntroductionDifferentialPower2011], differential fault analysis [@bihamDifferentialFaultAnalysis1997], fault sensitivity analysis [@liFaultSensitivityAnalysis2010a]).

![Relations between the different elements of an experiment bench.\label{fig:bench}](figures/experiment_bench.svg){width="60%"}

Side-channel and fault injection measurements are usually strongly impeded by noise. For side channel attacks, this implies that statistical tools must be used to increase the signal to noise ratio, requiring an important number of measurement samples. Fault injection campaigns usually boil down to using brute-force — or sometimes machine-learning techniques, when there is enough *a priori* knowledge on the properties of the search space [@picekEvolvingGeneticAlgorithms2014] — to find an adequate set of injection parameters in a large search space. In both cases, acquisition campaigns may last for weeks. This makes reliability and reproducibility important features of the data acquisition process. Additionally, as in other scientific fields, an experiment is only of lasting impact if it is transparent, can be reproduced and adapted to different situations.

Starting in 2008, different side-channel datasets have been made publicly available to researchers, either in the context of challenges or to provide reference traces for attacks evaluation. Among the most cited ones — DPA Contests [@DPAContests2008], *AES-HD* [@AES_HD2], *AES-RD* [@kizhvatovAES_RD2018], *ASCAD* [@prouffStudyDeepLearning2018; @ANSSIFRASCAD2018; @ASCADv22018], CHES 2023 challenge [@cassiersSMAesHchallengeFrameworkSidechannel2023] — only *AES_RD* provides the acquisition script used to produce the dataset. The other ones describe the acquisition setup, and give at most a concise natural language description of the acquisition procedure. This prevents a researcher from reproducing and extending the datasets, and provides no traceability for the collected datasets.

![Typical steps of the data acquisition process.\label{fig:experiment_process}](figures/acquisition_process.svg){width="100%"}

As illustrated by \autoref{fig:experiment_process}, a data acquisition script must usually generate a set of configurations covering the search space, and apply each of them to the instruments using manufacturer-provided or custom drivers. Then, measurements from some of the instruments are gathered and eventually stored in a permanent storage.

*Phileas* allows to iterate through complex configuration spaces in a way that
is easy to specify thanks to human readable configuration files. This
guarantees the repeatability of the experiments, and comes with the added
benefit of a standard description of the operations realized during an
acquisition campaign. While comparable tools exist, like Snakemake [@molderSustainableDataAnalysis2021] from bioinformatics or Prefect
[@PrefectHQPrefect2018] from machine learning, those are not really adapted to a
sequential context with high-dimension configuration spaces that must be
efficiently iterated over.

The test and measurement industry has come to some standardization of the
drivers, with for example the IVI foundation initiative
[@IVIFoundation]. However, only the most generic devices — like  oscilloscopes
and power supplies — are covered, leaving in particular all the specific
instrument that can be found in fault injection experiments — like glitch
generators, pulsed laser source and related optical instruments. These devices
are usually covered by dedicated drivers such as PyPDM [@LedgerHQPypdm2018] or
those of NewAE's ChipWhisperer[@NewaetechChipwhisperer2014] and ChipShouter
[@NewaetechChipSHOUTER2018], which are not designed with interoperability in
mind. Additionally, other standards like GCODE
[@kramerNISTRS274NGCInterpreter2000] cover the use of actuators like the
motorized stage used to modify the location of probes, albeit in a CNC-oriented
way which is not functionally appropriate, and would be too slow to be
effectively used. *Phileas* provides a soft standardization of the driving
interface of the instruments most commonly used in hardware security
experiments. This eases the replication of research works, but full compliance
is not enforced so as not to restrict the experimenter.

Finally, the SCA domain features a lot of open-source data analysis frameworks
[@boslandHighPerformanceDesignPatterns2024; @LedgerDonjonLascar2018; @GitHubEshardScared]) alongside some public databases([@ANSSIFRASCAD2018],
[@SMAesHChallengeOverview], [@AES_HD2],
[@clavierPracticalImprovementsSidechannel2014]). However, effort is lacking
toward the storage and use of the way that data is acquired. By specifying a
description of the experimental setup and process used to generate data, and
leveraging this to produce properly annotated *pandas* [@mckinney-proc-scipy-2010; @teamPandasdevPandasPandas2024] and xarray [@hoyer2017xarray] compatible data
outputs, *Phileas* provides researchers with a hassle-free method of producing
dataset that are self-describing, ensuring that they can be used for data
analysis straight out-of-the-box.

The lack of standardized acquisition tools fundamentally impedes scientific progress in hardware security research. Without reproducible methodologies, researchers cannot validate findings or build upon prior work, forcing them to reinvent infrastructure rather than advancing the science. *Phileas* addresses this critical gap by transforming hardware security research from an artisanal practice into a rigorous, collaborative science.

# References
