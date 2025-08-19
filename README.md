# Vision Transformer Experiments for Network Traffic Classification

## Purpose

This project explores applying Vision Transformers (ViT) to network security by converting packet payloads into visual representations for attack classification. The core innovation treats network security data as images, enabling computer vision techniques to identify attack patterns with up to 93.30% accuracy across 9 attack types.

## Research Paper

Full methodology and results are detailed in our research paper: [Vision Transformers for Network Traffic Classification](https://drive.google.com/file/d/18PNRFCJbhmeCsn_LwfRVt496Z7HSIWyk/view?usp=sharing)

## Overview

The research demonstrates that transforming network packet payloads into images allows sophisticated computer vision models to effectively detect cybersecurity threats. By treating bytes as pixel values and applying various encoding strategies, we achieve strong classification performance on real-world network traffic datasets.

## Dataset Processing

### CIC-IoT23 Dataset
The primary experiments use the CIC-IoT23 dataset, processed through `CIC-data-sampler.ipynb`:

- **Source**: CIC-IoT23 PCAP files from Google Cloud Storage
- **Sampling**: 12,000 samples per attack class
- **Payload Size**: First 1,500 bytes of each packet
- **Output**: Parquet files and PNG image samples
- **Organization**: Data structured by attack label for efficient training

### Attack Classes
The model classifies network traffic into 9 categories:
- Benign_Final (normal traffic)
- DDoS-HTTP_Flood
- DDoS-SYN_Flood
- DictionaryBruteForce
- DoS-TCP_Flood
- DoS-UDP_Flood
- Mirai-udpplain
- Recon-PortScan
- SqlInjection

## Image Encoding Methods

Five distinct strategies transform network payloads into visual representations:

1. **Grayscale 32×32**: Sequential byte mapping to single-channel images
2. **Grayscale 39×39**: Higher resolution grayscale with improved spatial detail
3. **RGB Hilbert 32×32**: Hilbert curve mapping preserving byte locality in RGB space
4. **RGB Spiral 32×32**: Center-outward spiral pattern encoding
5. **5-Channel 32×32**: Multi-dimensional representation with specialized channels:
   - Channel 1: Raw byte values
   - Channel 2: Header emphasis (first 64 bytes)
   - Channel 3: Byte frequency distribution
   - Channel 4: Local entropy measurements
   - Channel 5: Gradient magnitude calculations

## Model Architecture

All experiments use a consistent Vision Transformer configuration:
- **Patch Size**: 16×16 pixels
- **Patches per Image**: 4 (for 32×32 images)
- **Embedding Dimension**: 192-256
- **Attention Heads**: 3-4
- **Transformer Layers**: 6
- **Output Classes**: 9 attack categories

## Training Configuration

- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing
- **Loss Function**: Weighted cross-entropy for class imbalance handling
- **Data Split**: 70% training, 15% validation, 15% testing

## Results

### Performance Summary

| Encoding Method | Test Accuracy | Parameters | Key Characteristics |
|----------------|---------------|------------|-------------------|
| 5-Channel 32×32 | **93.30%** | 2.92M | Best performance; rich multi-dimensional features |
| RGB Hilbert 32×32 | 92.11% | 4.97M | Spatial locality preservation via Hilbert curve |
| Grayscale 39×39 | 91.01% | 2.71M | Higher resolution with improved detail |
| Grayscale 32×32 | 90.52% | 2.72M | Simple baseline approach |
| RGB Spiral 32×32 | 79.06% | 4.97M | Center-outward pattern encoding |

### Key Findings

**Multi-Channel Superiority**: The 5-channel encoding achieved the highest accuracy by incorporating multiple perspectives of payload data, demonstrating that diverse feature representations enhance classification performance.

**Spatial Encoding Impact**: RGB Hilbert encoding significantly outperformed RGB Spiral encoding, highlighting the importance of locality-preserving mappings in network data visualization.

**Model Efficiency**: The 5-channel approach achieved superior results with fewer parameters than RGB models through information-rich encoding strategies.

**Attack-Specific Performance**:
- Best classified: DoS-UDP_Flood (98.44%), DDoS-HTTP_Flood (97.72%)
- Most challenging: Recon-PortScan (86.33%), DictionaryBruteForce (87.78%)

### Common Misclassifications
- SqlInjection and DictionaryBruteForce due to similar payload patterns
- Recon-PortScan and Benign_Final as reconnaissance mimics normal behavior
- Strong distinction maintained between DDoS attack variants

## Initial Validation

The proof-of-concept validation on UNSW-NB15 dataset established feasibility:
- **Dataset**: 79,881 samples across 10 attack types
- **Architecture**: 39×39 grayscale images with 16×16 patches
- **Result**: 76.66% test accuracy
- **Significance**: Confirmed viability of treating network payloads as images

## Repository Structure

```
ViT-experiment/
├── CIC-data-sampler.ipynb               # Dataset generation from CIC-IoT23 PCAP
├── ViT_Prototype_Proof_of_Concept.ipynb # Initial UNSW-NB15 validation
├── ViT_Prototype_grayscale_32x32.ipynb  # Grayscale 32×32 experiments
├── ViT_Prototype_grayscale_39x39.ipynb  # Grayscale 39×39 experiments
├── ViT_Prototype_5channel_32x32.ipynb   # 5-channel experiments
├── ViT_Prototype_rgb_hilbert_32x32.ipynb # RGB Hilbert curve experiments
├── ViT_Prototype_rgb_spiral_32x32.ipynb  # RGB spiral experiments
├── results_*.json                        # Performance metrics
├── best_*_vit_model.pth                 # Trained model checkpoints
└── pcap-dataset-samples/                # Generated image datasets
```

## Requirements

- PyTorch >= 1.9.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Google Cloud Storage access
- CUDA-capable GPU (recommended) or 8-16 GB RAM

## Future Research Directions

- **Encoding Optimization**: Develop learned encoding strategies
- **Architecture Scaling**: Evaluate larger ViT models and patch configurations
- **Real-time Deployment**: Optimize models for production inference
- **Explainability**: Implement attention visualization for attack signature identification

## Conclusion

This research establishes that Vision Transformers can effectively classify network traffic through visual payload representation. The 5-channel encoding approach achieving 93.30% validation accuracy demonstrates the potential of computer vision techniques in cybersecurity applications, providing a foundation for future visual-based network security analysis.