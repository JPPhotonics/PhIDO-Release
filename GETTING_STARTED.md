# Getting Started with PhIDO

This comprehensive guide will walk you through setting up and using PhIDO (Photonics Intelligent Design and Optimization) for automated photonic circuit design. PhIDO offers two distinct workflow modes to accommodate different user preferences and use cases.

## Table of Contents

1. [Prerequisites and Installation](#prerequisites-and-installation)
2. [Configuration](#configuration)
3. [Running PhIDO](#running-phido)
4. [Automatic Workflow (Guided Mode)](#automatic-workflow-guided-mode)
5. [Step-by-Step Workflow (Independent Mode)](#step-by-step-workflow-independent-mode)
6. [Practical Examples](#practical-examples)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Prerequisites and Installation

### System Requirements

- **Operating System**: Linux (Ubuntu/Debian recommended)
- **Python**: 3.12 or higher
- **Memory**: At least 8GB RAM recommended
- **Storage**: At least 2GB free space

### Step 1: Install System Dependencies

First, install the required system packages:

```bash
# Update package list
sudo apt-get update

# Install essential dependencies
sudo apt-get install graphviz libgraphviz-dev pkg-config klayout
sudo apt-get install -y build-essential python3-dev swig
```

### Step 2: Set Up Python Environment

```bash
# Navigate to the PhIDO directory
cd /path/to/PhIDO-Release

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install kfactory (required for layout generation)
pip install kfactory==0.21.1
```

**Note**: Ignore any pip dependency resolver conflicts with gdsfactory 8.8.5 - this is expected.

### Step 4: Create Required Directories

```bash
# Create log directory
mkdir -p PhotonicsAI/log
```

### Step 5: Set Environment Variables

```bash
# Set Python path (required for imports)
export PYTHONPATH='.'
```

## Configuration

### API Keys Setup

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env
```

Add your API keys to the `.env` file:

```bash
# Required for OpenAI models (GPT-4, GPT-4o, O1)
OPENAI_API_KEY='your-openai-api-key'

# Required for Anthropic Claude models
ANTHROPIC_API_KEY='your-anthropic-api-key'

# Required for Google Gemini models
GOOGLEGENAI_API_KEY='your-google-api-key'

# Required for DeepSeek-R1
DEEPSEEK_API_KEY='your-deepseek-api-key'

# Required for models hosted on NVIDIA NIM
NVIDIA_API_KEY='your-nvidia-nim-api-key'
```

**Important**: The `OPENAI_API_KEY` is required even if you're using other models, as PhIDO uses GPT models for formatting entity extraction results.

### Supported LLM Models

PhIDO supports multiple LLM providers. You can configure different models for different steps:

#### OpenAI Models
- **GPT-4o** (`gpt-4o`) - General purpose reasoning
- **o1** (`o1`) - Specialized for reasoning tasks (Default)
- **o3-mini** (`o3-mini`) - Faster reasoning model

#### Anthropic Models
- **Claude-3-7-Sonnet** (`claude-3-7-sonnet-latest`) - Balanced performance
- **Claude-4.0-Opus** (`claude-opus-4-20250514`) - Advanced reasoning

#### Google Models
- **Gemini-2.5-Pro** (`gemini-2.5-pro`) - Advanced reasoning
- **Gemini-1.5-Pro** (`gemini-1.5-pro`) - General purpose
- **Gemini-1.5-Flash** (`gemini-1.5-flash`) - Fast response
- **Gemini-2.0-Flash** (`gemini-2.0-flash`) - Latest flash model

#### NVIDIA Models (via NIM API)
- **Llama-3.1-Nemotron-Ultra** (`nvidia/llama-3.1-nemotron-ultra-253b-v1`)
- **Nemotron-4-340B** (`nvidia/nemotron-4-340b-instruct`) - Deprecated

#### DeepSeek Models
- **DeepSeek-Reasoner** (`deepseek-reasoner`) - Specialized reasoning

## Running PhIDO

### Quick Start

```bash
# Using Makefile (recommended)
make run
```

### Manual Start

```bash
# Ensure environment is set up
export PYTHONPATH='.'
source venv/bin/activate

# Start the application
streamlit run PhotonicsAI/Photon/webapp.py
```

The application will start and be available at `http://localhost:8501` in your web browser if port is available, see message in terminal for actual port address if port 8501 is busy on your system.

## Automatic Workflow (Guided Mode)

The Automatic Workflow provides a fully automated, step-by-step process that guides users through the entire circuit design pipeline with minimal user intervention.

### Overview

The Automatic Workflow consists of 5 main phases:
1. **Input Processing** - User provides circuit description
2. **Entity Extraction** - AI analyzes and extracts circuit requirements
3. **Component Selection** - AI matches components from knowledge base
4. **Schematic Generation** - Creates circuit DSL and schematic
5. **Layout & Simulation** - Generates GDS files and runs simulations
6. **Design Rule Checking** - Validates layout against manufacturing rules

### Step-by-Step Guide

#### Step 1: Start the Application

1. Open your web browser and navigate to `http://localhost:8501` (or whichever port streamlit has set up PhIDO webapp in.)
2. You'll see the PhIDO interface with workflow mode selection
3. Select **"Automatic"** from the workflow mode dropdown

#### Step 2: Enter Circuit Description

1. In the text input area, describe your photonic circuit requirements
2. Be as specific as possible about:
   - Circuit type (e.g., "Mach-Zehnder interferometer")
   - Components needed (e.g., "with TiN heaters")
   - Specifications (e.g., "2x2 configuration", "C-band operation")
   - Performance requirements

**Example Input:**
```
Design a 2x2 Mach-Zehnder interferometer with TiN heaters for C-band operation. 
The circuit should have input/output ports and be suitable for optical switching applications.
```

#### Step 3: Entity Extraction (Automatic)

1. Click **"Submit"** or press Enter
2. PhIDO will automatically:
   - Analyze your input using the selected LLM
   - Extract circuit requirements and specifications
   - Identify required components and parameters
   - Generate a structured pretemplate

**What happens behind the scenes:**
- The system uses AI to parse your natural language description
- Extracts technical specifications (wavelength, dimensions, materials)
- Identifies circuit topology and component requirements
- Creates a structured YAML pretemplate for the next phase

#### Step 4: Component Selection (Automatic)

1. PhIDO automatically searches the knowledge base for matching components
2. The system will:
   - Match extracted requirements with available photonic components
   - Search the DesignLibrary for suitable components
   - Present component options with descriptions and references
   - Allow you to select preferred components if multiple options exist

**Component Selection Interface:**
- Review suggested components
- Select your preferred options from available alternatives
- Click **"Submit Component Selection"** to proceed

#### Step 5: Schematic Generation (Automatic)

1. PhIDO automatically generates the circuit schematic
2. The system will:
   - Create a circuit DSL (Domain Specific Language) representation
   - Generate component connections and routing
   - Create a visual schematic diagram
   - Prepare the circuit for layout generation

**Schematic Output:**
- Visual circuit diagram showing component connections
- Circuit DSL in YAML format
- Component specifications and parameters
- Port definitions and signal flow

#### Step 6: Layout & Simulation (Automatic)

1. PhIDO automatically generates the physical layout
2. The system will:
   - Create GDS layout files for fabrication
   - Generate component geometries and routing
   - Run circuit simulations using SAX
   - Provide performance analysis results

**Layout Output:**
- GDS files ready for fabrication
- Visual layout representation
- Simulation results and performance metrics
- Component placement and routing information

#### Step 7: Design Rule Checking (Automatic)

1. PhIDO automatically validates the layout
2. The system will:
   - Run KLayout DRC (Design Rule Checking)
   - Check for manufacturing rule violations
   - Generate DRC reports with error details
   - Provide recommendations for fixes if needed

**DRC Results:**
- Pass/Fail status for manufacturing readiness
- Detailed violation reports
- Visual highlighting of problem areas
- Suggestions for design improvements

### Automatic Workflow Features

- **Guided Progression**: Automatic advancement between phases
- **Real-time Feedback**: Live status updates and progress indicators
- **Integrated Validation**: Built-in checks at each stage
- **Template-based Generation**: Uses predefined circuit templates

### Tips for Automatic Workflow

1. **Be Specific**: Provide detailed circuit descriptions for better results
2. **Use Technical Terms**: Include wavelength bands, component types, and specifications

## Step-by-Step Workflow (Independent Mode)

The Step-by-Step Workflow allows users to execute individual workflow steps with custom inputs and manual control over each phase.

### Overview

The Step-by-Step Workflow provides 5 independent execution steps:
1. **Entity Extraction** - Run with custom circuit descriptions
2. **Component Specification** - Execute with predefined templates or custom components
3. **Circuit DSL Creation** - Generate circuit DSL from custom inputs
4. **Schematic Generation** - Create schematics independently
5. **Layout & Simulation** - Generate layouts and run simulations

### Step-by-Step Guide

#### Step 1: Start the Application

1. Open your web browser and navigate to `http://localhost:8501`
2. Select **"Step-by-Step"** from the workflow mode dropdown
3. You'll see a step selection interface

#### Step 2: Select Execution Step

Choose from the available steps:
- **Entity Extraction** - Extract requirements from text
- **Component Specification** - Search and select components
- **Circuit DSL Creation** - Create circuit representation
- **Schematic Generation** - Generate visual schematics
- **Layout & Simulation** - Create layouts and run simulations

#### Step 3: Entity Extraction

**Purpose**: Extract circuit requirements from natural language descriptions

**How to use:**
1. Select "Entity Extraction" from the step dropdown
2. Enter your circuit description in the text area
3. Click **"Run Entity Extraction"**

**Example Input:**
```
Create a 1x1 Mach-Zehnder interferometer with TiN heaters for C-band operation.
The device should have 500nm wide waveguides and 10um long heater elements.
```

**Output:**
- Structured pretemplate in YAML format
- Extracted circuit specifications
- Component requirements
- Parameter definitions

**Customization Options:**
- Modify the input description
- Adjust extraction parameters
- Review and edit extracted specifications

#### Step 4: Component Specification

**Purpose**: Search and select photonic components from the knowledge base

**How to use:**
1. Select "Component Specification" from the step dropdown
2. Choose input method:
   - **Use Previous Step**: Use entity extraction results
   - **Custom Pretemplate**: Provide your own YAML pretemplate
   - **Template Selection**: Choose from predefined templates
3. Click **"Run Component Search"**

**Template Options:**
- **MZI with TiN heaters** - Mach-Zehnder interferometer with thermal tuning
- **2x2 MZI with heaters** - 2x2 configuration with multiple heaters
- **Directional coupler** - Basic coupling device
- **Custom templates** - User-defined configurations

**Component Selection Interface:**
- Review search results
- Select preferred components from alternatives
- Choose templates for circuit generation
- Customize component parameters

**Output:**
- Selected component list
- Template specifications
- Component parameters and descriptions
- Reference documentation links

#### Step 5: Circuit DSL Creation

**Purpose**: Generate circuit DSL (Domain Specific Language) representation

**How to use:**
1. Select "Circuit DSL Creation" from the step dropdown
2. Choose input method:
   - **Use Previous Step**: Use component specification results
   - **Custom Pretemplate**: Provide your own YAML
   - **Manual Component List**: Specify components manually
3. Configure circuit parameters
4. Click **"Create Circuit DSL"**

**Parameter Configuration:**
- Component dimensions (length, width, height)
- Material properties (refractive index, conductivity)
- Operating parameters (wavelength, temperature)
- Connection specifications (ports, routing)

**Output:**
- Circuit DSL in YAML format
- Component definitions and connections
- Port specifications
- Parameter assignments

#### Step 6: Schematic Generation

**Purpose**: Create visual circuit schematics and diagrams

**How to use:**
1. Select "Schematic Generation" from the step dropdown
2. Provide circuit DSL input:
   - **Use Previous Step**: Use DSL from previous step
   - **Custom Circuit DSL**: Provide your own YAML
3. Optionally provide custom preschematic
4. Click **"Run Schematic Generation"**

**Schematic Options:**
- **Auto-generated**: Let AI create the schematic
- **Custom preschematic**: Provide your own schematic description
- **Template-based**: Use predefined schematic templates

**Output:**
- Visual circuit diagram
- Component layout and connections
- Port definitions
- Signal flow representation

#### Step 7: Layout & Simulation

**Purpose**: Generate physical layouts and run circuit simulations

**How to use:**
1. Select "Layout & Simulation" from the step dropdown
2. Provide circuit DSL input:
   - **Use Previous Step**: Use DSL from previous step
   - **Custom Circuit DSL**: Provide your own YAML
3. Click **"Run Layout & Simulation"**

**Simulation Options:**
- **SAX Simulation**: Circuit performance analysis
- **FDTD Integration**: Electromagnetic field simulation (available only on the `tidy3d_integration` branch)

**Output:**
- GDS layout files
- Visual layout representation
- Simulation results and metrics
- Performance analysis data

### Step-by-Step Workflow Features

- **Manual Control**: Execute each step independently
- **Custom Inputs**: Provide custom data at each step
- **Detailed Inspection**: Review results at each phase
- **Flexible Workflow**: Skip or repeat steps as needed
- **Template Support**: Use predefined or custom templates
- **Export Capabilities**: Save intermediate and final results

### Tips for Step-by-Step Workflow

1. **Start Simple**: Begin with basic circuits to understand the process
2. **Use Templates**: Leverage predefined templates for common circuits
3. **Save Intermediate Results**: Export data between steps for backup
4. **Experiment with Parameters**: Try different configurations
5. **Review Each Step**: Check outputs before proceeding to next step
6. **Combine Steps**: Use results from one step as input to another

## Practical Examples

The following examples demonstrate the Step-by-Step Workflow with increasing complexity levels. **Note**: These examples can also be run using the Automatic Workflow, but the intermediate step outputs and manual control shown here would not be applicable in automatic mode.

For example outputs for each intermediate stage for all 4 prompts, refer to `./GETTING_STARTED_EXAMPLE_OUTPUTS`.

### Example 1: Level 1 - Single Component with Specifications

**Objective**: Create a simple directional coupler with specific parameters

**Input Prompt:**
```
Design a directional coupler with 50/50 splitting ratio for C-band operation. 
The coupler should have 500nm wide waveguides, 2um coupling length, and 200nm gap between waveguides.
```

**Step-by-Step Execution:**

1. **Entity Extraction**:
   - Input the prompt above
   - Review extracted specifications: wavelength (C-band), splitting ratio (50/50), dimensions (500nm width, 2um length, 200nm gap)
   - Verify component type: directional coupler

2. **Component Specification**:
    - Input the entity extraction output into the entry field (should already be autopopulated for you).
   - LLM search and retrieve for directional coupler components
   - Select appropriate coupler from DesignLibrary
   - Review component parameters and documentation

3. **Circuit DSL Creation**:
   - Use selected component with specified parameters
   - Both `Pretemplate YAML` and `Custom Selected Components` input fields should be autopopulated for you with the entity extraction and component specification outputs respectively.
   - Designer agent will define input/output ports
   - Set coupling length and gap specifications

4. **Schematic Generation**:
   - Generate visual representation of the coupler
   - DSL YAML should be autopopulated for you if you ran the DSL creation step.
   - The designer agent here verifies port connections and dimensions and generates a detailed schematic DOT graph.

5. **Layout & Simulation**:
   - Create GDS layout with specified dimensions
   - Run optical simulation for C-band performance
   - Verify 50/50 splitting ratio at target wavelength

**Expected Output:**
- Single directional coupler component
- GDS layout with 500nm waveguides and 200nm gap
- Simulation results showing 50/50 splitting at C-band

### Example 2: Level 2 - Two Components Connected

**Objective**: Create two cascaded Mach-Zehnder interferometers with high-speed modulation capability

**Input Prompt:**
```
Two cascaded MZIs, each with a modulation bandwidth up to 10 GHz. Both MZIs have two input/output ports.
```

**Step-by-Step Execution:**

1. **Entity Extraction**:
   - Extract system specifications: 2 cascaded MZIs, 10 GHz modulation bandwidth
   - Identify component configuration: 2x2 MZIs (two input/output ports each)
   - Determine performance requirements: high-speed modulation capability

2. **Component Specification**:
   - Search for high-speed MZI components with 2x2 configuration
   - Select MZI components suitable for 10 GHz modulation
   - Choose components with appropriate bandwidth specifications

3. **Circuit DSL Creation**:
   - Define two MZI components in cascade configuration
   - Specify 2x2 port configuration for each MZI
   - Set modulation bandwidth requirements (10 GHz)
   - Create interconnections between the two MZIs

4. **Schematic Generation**:
   - Create cascaded MZI schematic
   - Show 2x2 port configuration for each MZI

5. **Layout & Simulation**:
   - Generate layout with proper MZI cascading
   - Run high-frequency simulations for 10 GHz bandwidth
   - Analyze modulation performance and signal integrity
   - Verify cascaded system performance

**Expected Output:**
- Two cascaded 2x2 MZIs
- Layout optimized for high-speed operation
- Simulation results showing 10 GHz modulation bandwidth

### Example 3: Level 3 - Multiple Components (1x8 MZI Splitter Tree)

**Objective**: Create a 1x8 power splitter using cascaded MZI stages

**Input Prompt:**
```
Design a 1x8 power splitter using a three-stage MZI tree structure. 
The first stage has 1 MZI splitting into 2 outputs, the second stage has 2 MZIs splitting to 4 outputs, and the third stage has 4 MZIs splitting to 8 outputs. Each MZI should have 50/50 splitting ratio and 500nm wide waveguides for C-band operation.
```

**Step-by-Step Execution:**

1. **Entity Extraction**:
   - Extract tree structure: 3 stages, 1→2→4→8 outputs
   - Identify component count: 7 MZIs total (1+2+4)
   - Specify splitting ratio: 50/50 for all MZIs
   - Set waveguide specifications: 500nm width, C-band

2. **Component Specification**:
   - Search for MZI components suitable for splitting
   - Select 1x2 MZI template with 50/50 splitting
   - Plan component arrangement for tree structure

3. **Circuit DSL Creation**:
   - Define hierarchical structure: Stage 1 (1 MZI), Stage 2 (2 MZIs), Stage 3 (4 MZIs)
   - Create interconnections between stages
   - Specify input port and 8 output ports
   - Set equal power distribution requirements

4. **Schematic Generation**:
   - Generate tree structure schematic
   - Show cascaded MZI connections
   - Display power distribution network
   - Verify 1x8 splitting topology

5. **Layout & Simulation**:
   - Create compact tree layout
   - Optimize routing between stages
   - Run simulation for power uniformity across 8 outputs
   - Analyze insertion loss and crosstalk

**Expected Output:**
- 7 MZIs arranged in 3-stage tree
- 1 input and 8 output ports
- Simulation showing equal power distribution

### Example 4: Level 4 - Complex Multi-Stage System

**Objective**: Create a sophisticated 1x16 power splitter with integrated VOAs and phase shifters

**Input Prompt:**
```
Design a 1×16 power splitter using 15 1×2 MMIs in a four-stage tree: the first stage has 1 MMI splitting into 2 outputs, 
the second stage has 2 MMIs splitting to 4 outputs, and so on, until 16 outputs. Each of the 16 outputs should feed a VOA 
(variable optical attenuator) and then a thermo-optic phase shifter, finally connecting to a grating coupler.
```

**Step-by-Step Execution:**

1. **Entity Extraction**:
   - Extract complex system: 4-stage MMI tree (1→2→4→8→16)
   - Identify component count: 15 MMIs + 16 VOAs + 16 phase shifters + 16 grating couplers
   - Specify component types: MMI splitters, VOAs, thermo-optic phase shifters, grating couplers
   - Set operating wavelength and performance requirements

2. **Component Specification**:
   - Search for MMI 1x2 splitters with 50/50 ratio
   - Select VOA components with variable attenuation
   - Choose thermo-optic phase shifters with TiN heaters
   - Find grating couplers for fiber coupling
   - Plan component integration and routing

3. **Circuit DSL Creation**:
   - Define 4-stage MMI tree structure (1+2+4+8=15 MMIs)
   - Add 16 VOAs connected to each output
   - Integrate 16 thermo-optic phase shifters after VOAs
   - Connect 16 grating couplers as final outputs
   - Specify control signals for VOAs and phase shifters

4. **Schematic Generation**:
   - Create comprehensive system schematic
   - Show complete signal path from input to 16 outputs
   - Display control connections for active components
   - Illustrate power distribution and phase control capability

5. **Layout & Simulation**:
   - Generate complex multi-layer layout
   - Optimize routing for 16 parallel channels
   - Run system-level simulation for power uniformity
   - Analyze VOA and phase shifter performance
   - Verify grating coupler coupling efficiency

**Expected Output:**
- 15 MMI splitters in 4-stage tree
- 16 VOAs with individual control
- 16 thermo-optic phase shifters
- 16 grating couplers for fiber coupling
- Complete system simulation results

### Tips for Complex Examples

1. **Start Simple**: Begin with Level 1 examples to understand the workflow
2. **Incremental Complexity**: Progress through levels to build understanding
3. **Parameter Tuning**: Adjust component specifications between steps
4. **Result Validation**: Check outputs at each step before proceeding
5. **Export Intermediate Results**: Save DSL and schematics for reference
6. **Performance Analysis**: Review simulation results for each complexity level

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when starting PhIDO

**Solution**:
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH='.'

# Check virtual environment activation
source venv/bin/activate

# Verify installation
pip list | grep -E "(streamlit|gdsfactory|kfactory)"
```

#### 2. KLayout DRC Failures

**Problem**: Design Rule Checking fails or produces errors

**Solution**:
```bash
# Verify KLayout installation
klayout -v

# Check DRC script permissions
ls -la PhotonicsAI/Photon/drc/drc_script.drc

# Test with simple layout
klayout -b -r PhotonicsAI/Photon/drc/drc_script.drc
```

#### 3. LLM API Errors

**Problem**: API calls fail or return errors

**Solution**:
```bash
# Check API keys in .env file
cat .env | grep API_KEY

# Test API connectivity
python -c "import openai; print('OpenAI API key configured')"

# Verify model availability
# Check provider-specific documentation for model status
```

#### 4. GDS Layer Number Errors

**Problem**: Large layer numbers cause issues

**Solution**:
- PhIDO automatically handles layer number limitations
- Check GDS file with KLayout viewer
- Verify layer definitions in component files

#### 5. Memory Issues

**Problem**: Application runs out of memory during layout generation

**Solution**:
```bash
# Monitor memory usage
htop

# Increase swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 6. pygraphviz Installation Issues

**Problem**: pygraphviz fails to build

**Solution**:
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y build-essential python3-dev swig

# Reinstall pygraphviz
pip uninstall pygraphviz
pip install pygraphviz
```

### Debug Mode

Enable debug mode for detailed error information:

1. **Session State Debug**: Use the debug expander in step-by-step mode
2. **Log Files**: Check `PhotonicsAI/log/` directory for detailed logs
3. **Console Output**: Monitor terminal output for error messages
4. **Network Tab**: Check browser developer tools for API call issues

### Performance Optimization

1. **Model Selection**: Use faster models (GPT-4o, Gemini-Flash) for development
2. **Batch Processing**: Process multiple circuits in sequence
3. **Template Caching**: Reuse templates for similar circuits
4. **Resource Monitoring**: Track memory and CPU usage

## Advanced Usage

### Custom Component Development

1. **Create Component File**: Add new components to `KnowledgeBase/DesignLibrary/`
2. **Define Geometry**: Specify component dimensions and parameters
3. **Add S-Parameters**: Include simulation models
4. **Update Templates**: Add to `templates.yaml`
5. **Test Integration**: Verify with PhIDO workflows

### DRC Rule Customization

1. **Modify DRC Script**: Edit `drc/drc_script.drc`
2. **Add New Rules**: Define custom design rules
3. **Update Layer Definitions**: Modify layer constraints
4. **Test Rules**: Validate with sample layouts

### Workflow Automation

1. **Script Integration**: Use PhIDO programmatically
2. **Batch Processing**: Process multiple designs
3. **API Integration**: Connect with external tools
4. **Custom Templates**: Create domain-specific templates

### Performance Tuning

1. **Model Optimization**: Select appropriate LLM models for each step
2. **Caching**: Implement result caching for repeated operations
3. **Parallel Processing**: Run multiple simulations simultaneously
4. **Resource Management**: Optimize memory and CPU usage

---

## Getting Help

- **Documentation**: Check the main README.md for detailed information
- **Issues**: Report bugs and request features on the project repository
- **Community**: Join discussions and share experiences
- **Examples**: Explore the Testbench.xlsx for 102 testbench prompts

## Next Steps

1. **Explore Templates**: Try different circuit templates
2. **Experiment with Parameters**: Modify component specifications
3. **Create Custom Circuits**: Design your own photonic circuits
4. **Integrate with Tools**: Connect PhIDO with your existing workflow
5. **Contribute**: Help improve PhIDO by contributing code or documentation

Happy designing with PhIDO! 🚀
