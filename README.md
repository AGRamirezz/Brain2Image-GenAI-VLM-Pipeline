# Brain-to-Image: EEG-to-Visual Reconstruction Pipeline

> Reconstructing visual stimuli from EEG signals using modern Vision Language Models

## Overview

This project demonstrates a novel approach to brain-computer interface technology by reconstructing visual images from EEG brain signals. Using an Emotiv 5-channel Insight headset, we capture neural responses to ImageNet visual stimuli and reconstruct the viewed images through a multi-stage AI pipeline.

## 🧠 How It Works

### Pipeline Architecture

1. **EEG Signal Capture**
   - Emotiv 5-channel Insight EEG headset
   - Participants view ImageNet animal stimuli
   - Real-time neural signal acquisition

2. **Neural Classification** 
   - Pre-trained 2D CNN classifies EEG signals
   - Outputs predicted animal category (goldfish, mantaray, rooster, ostrich)

3. **Image Reconstruction**
   - CNN-based model reconstructs noisy pixel predictions from EEG
   - Creates initial "messy" visual representation

4. **VLM Enhancement Pipeline** ⭐ *New Approach*
   - Vision Language Model (BLIP-2) analyzes noisy reconstruction + category
   - Generates detailed scene description and context
   - Stable Diffusion performs image-to-image refinement
   - Produces high-quality final reconstruction

## 🔬 Technical Stack

- **Hardware**: Emotiv 5-channel Insight EEG headset
- **Deep Learning**: PyTorch, 2D CNNs for EEG processing
- **Vision Language Models**: BLIP-2, Transformers
- **Image Generation**: Stable Diffusion, Diffusers
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**: Matplotlib, PIL

## 🚀 Key Features

- **Modern VLM Approach**: Leverages latest vision-language models for intelligent reconstruction
- **End-to-End Pipeline**: Complete workflow from EEG signals to final images
- **Modular Design**: Easy integration of custom EEG models
- **Batch Processing**: Support for multiple subjects and sessions
- **Comprehensive Analysis**: Visualization and performance metrics
- **Research Ready**: Built for experimentation and iteration

## 📊 Current Results

The project includes reconstructions for 4 animal categories:
- 🐠 Goldfish (2 variants)
- 🦅 Mantaray (2 variants)  
- 🐓 Rooster (2 variants)
- 🦢 Ostrich (2 variants)

Each category shows progression from original stimulus → noisy EEG reconstruction → VLM-enhanced final image.

## 🔧 Getting Started

### Prerequisites

```bash
pip install torch transformers diffusers pillow matplotlib pandas opencv-python
```

### Quick Start

1. **Load the pipeline**:
   ```python
   from brain2image_vlm_pipeline_template import brain_to_image_pipeline
   ```

2. **Process EEG data**:
   ```python
   results = brain_to_image_pipeline(eeg_data, subject_id="test_001")
   ```

3. **Visualize results**:
   ```python
   visualize_pipeline_results(results)
   ```

## 🎯 Next Steps

### Immediate Improvements
- [ ] Integrate actual pre-trained EEG models
- [ ] Implement proper EEG data loading utilities
- [ ] Optimize VLM prompt engineering for better reconstructions
- [ ] Add evaluation metrics (CLIP score, FID, perceptual similarity)

### Research Extensions
- [ ] Real-time processing capabilities
- [ ] Cross-subject generalization studies
- [ ] Extended stimulus categories beyond animals
- [ ] Attention visualization and analysis
- [ ] Multi-modal VLM experimentation (GPT-4V, LLaVA)

### Technical Enhancements
- [ ] Model ensemble methods
- [ ] Advanced EEG preprocessing pipelines
- [ ] Custom VLM fine-tuning on domain data
- [ ] Performance optimization and GPU acceleration

## 📚 Research Context

This work builds upon recent advances in:
- **Neural Decoding**: Converting brain signals to meaningful outputs
- **Vision Language Models**: Multimodal AI for image understanding
- **Generative AI**: Modern diffusion models for image synthesis
- **Brain-Computer Interfaces**: Non-invasive EEG-based systems

## 🔗 References

*To be completed*

---

**Note**: This project is under active development. The pipeline currently uses placeholder models that should be replaced with your trained EEG classification and reconstruction models for full functionality.
