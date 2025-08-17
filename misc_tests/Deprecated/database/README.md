# Photonic Knowledge Base

We are building a database of photonic components.

### scope
To make the problem tractable at this early stage, the scope of this database is limited to
- silicon photonics
- for active components, like modulators, we could ignore electrical connections
- only telecom o-band
- both polarizations

### database format
I think a simple tree database is not good because photonic elements and their functional space is more complex than a single tree. Depending on the problem we want to solve, a human designer would reshuffle their mental tree. For example, a MMI can be structured in different ways depending on the context:
- Passive -> silicon photonics -> power splitter -> MMI
- silicon photonics -> 1x2 -> dichroic filter -> MMI
- silicon photonics -> inverse design -> MMI

The database is written in **yaml**, which is the most human readable format I can think of. This can be easily ported into sql or neo4j. Neo4j graph database looks like a very good candidate for the final implementation. to keep our yaml file compatible with neo4j, we need to format everything as following
- nodes
	- create a node for each element
	- can have many {labels} and many {properties}
	- neo4j is optimized to query {labels} very fast
		- ``` MATCH (n:silicon:passive) RETURN n ```
		- ``` MATCH (n {name: 'MMI-1'}) RETURN n.description, n.sMatrixAddress ```
	- labels cannot contains spaces, instead we should use snake_case
	- properties are the info stored inside the node. these include the s-matrices, python/yaml code, params or input arguments

- relationships
	- can have one {type} and many {properties}
	- simplest way of representing a circuit:
		- create a relationship for each photonic connection
		- set the {type} to the name of the circuit
		- add details of the connection (like port numbers etc) as relationship properties

Example:
```
nodes:
  - nodeLabels: [silicon, passive, power_splitter, 2x2, narrowband, dichroic_filter]
    nodeProperties:
      name: MMI-1
      description: multimode interferometer
      pythonClass: |
        def mmi1x2_wavelength_1530_to_1565():
        return gf.components.mmi1x2(width_taper=1.4, length_taper=10.0, length_mmi=12.8,
                                width_mmi=3.8, gap_mmi=0.25, width=0.0, cross_section='xs_sc')
      yamlCode: |
        component: mmi2x2
        settings:
            width_taper: 1
            length_taper: 5
            length_mmi: 5
            width_mmi: 2.5
            gap_mmi: 0.50
      params: wavelength, polarization
      band: o-band
      loss: 2 dB
      sMatrixAddress: ./s-matricies/mmi-1/s.dat
relationships:
  - relationshipType: 400G_FR4_transceiver
    source: MZM-1
    destination: CWDM-1
    relationshipProperties:
      source_port: 2
      destination_port: 1
```

### parameterized components vs different versions of a component
<img src="./img/turtles.jpeg" width="300">

 We can create multiple versions of a component like MMI-1 and MMI-2. This should only happen if these two belong to a different set of {labels}, like forward and reverse design MMIs or Si and SiN MMIs.

Almost all components will be parameterized, i.e. they can adopt their functionality depending on a few input arguments. From a human designer perspective, these parameters are things like design wavelength, polarization, etc. On the otherhand, the python functions in gdsfactory accept arguments like device length and width. So to implement the 'params' in our database, we need an abstraction level.

For example, we want to store an MMI with a ten different design wavelengths (across o-band) and for TE/TM polarizations in the database.
- params => 10x design_wavelength, 2x polarization => calculate 20 s_matrices and save 20 interconnect files with descriptive filenames
- we also need a yaml file with a list of available {params}, the arguments used to calculate each, and the s_matrix uri


### open questions
- for nodes, what labels should we use? one way to think about this is a rather high level types of components that lives in the head of a photonic engineer. maybe:
	- silicon_photonics (assigned programmatically)
	- NxM (assigned programmatically)
	- passive
	- active
	- marker (text, geometrical shapes, cross, varioud litho markers, )
 	- inverse_design
	- narrowband? (assigned programmatically)
	- broadband? (assigned programmatically)
  	- wire (including straight and bend waveguides, tapers, crossing, delay_line, spiral, cutback)
	- power_splitter
	- wavelength_operator (filter, mux, converter: resonator, MMI, MZI)
 	- mode_operator (both polarization and spatial mode operators like filter, mux, converter: PBS, te-tm converter, te0-te1 converter, )
	- modulator (amplitude, phase, polarization)
 	- resonator
  	- chip_coupler (grating coupler, edge coupler, mode converters, )
  	- detector
  	- laser
  	- heater
  	- directional_coupler (sym, asym, adiabatic, pully, )
  	- grating? (bragg_reflector, )
  	- if we have the s-matrices with their meta data, we could programmatically generate many of these labels

- what are the component properties we want to store in the database. maybe:
	- name
 	- description
  	- pythonClass
  	- yamlCode
  	- band (currently only o-band)
  	- params
  	- sMatrixAddress
- Many devices, can inherently be a single component or a circuit. For example a conventional MMI is a single component but we can also build a circuit of many dicrectional couplers providing very similar functionality but broadband. Or WDMs, modulators etc. Does this require any special consideration? or not?
- The most simple way to represent a circuit is to define it only in the relationships. But we could add it as a new node:
	- how to define the photonic connections?
	- this is probably more complicated, what's the advantage?
