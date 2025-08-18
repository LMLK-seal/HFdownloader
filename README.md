# HFdownloader.com

![HFdownloader logo](https://github.com/LMLK-seal/HFdownloader/blob/main/logo.png?raw=true)

[![Website](https://img.shields.io/website?url=https%3A//hfdownloader.com)](https://hfdownloader.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **[HFdownloader.com](https://hfdownloader.com)** is a streamlined web application for downloading entire Hugging Face model repositories as single ZIP files.

## 🚀 Features

- **Simple Interface**: Clean, intuitive design for effortless model downloads
- **Complete Repository Download**: Download entire model repositories, including all files and configurations
- **Progress Tracking**: Real-time download progress with detailed status logs
- **ZIP Compression**: All models packaged as convenient ZIP archives
- **Wide Model Support**: Compatible with thousands of models from the Hugging Face Hub
- **No Installation Required**: Browser-based tool - no software installation needed

## 🌐 Live Website

Visit **[HFdownloader.com](https://hfdownloader.com)** to start downloading models instantly.

## 📋 How to Use

1. **Enter Model ID**: Input the Hugging Face model identifier (e.g., `unsloth/Llama-3.2-3B-Instruct`)
2. **Click Download**: Press the "Start Download" button to begin
3. **Monitor Progress**: Watch the real-time progress and status updates
4. **Receive ZIP File**: Your complete model repository will be delivered as a ZIP archive
- `Note: Models that require accepting license terms on the Hugging Face Hub may fail to download.`

## 💡 ChatBot

- I've added a chatbot that loads a downloaded model folder and lets you chat with an AI.
```bash
# Install required packages
pip install customtkinter tkinter pathlib configparser
```
1. **Load Your Model:**: Type your message into the input box at the bottom and press Enter.
2. **Start Chatting (Optional):** Use the "Load File (TXT/PDF)" button to add context from a document to your conversation.
3. **Set GPU Offload (Optional):**
- Before loading a model, look at the "GPU Offload" setting on the right. This lets you run large models by splitting them between your fast GPU and your computer's regular RAM.
- -1 (Auto - Recommended): The app automatically puts as many layers as possible onto your GPU. This is the best option to try first.
- A positive number (e.g., 20): Manually set how many layers to load onto the GPU. Use this to fine-tune memory usage.
- 0: Runs the entire model on the CPU. This is the slowest option, but works if you don't have a powerful GPU.

## 🎯 Use Cases

- **Offline Development**: Download models for offline machine learning development
- **Backup Solutions**: Create local backups of important model repositories
- **Educational Purposes**: Access model files for learning and research
- **Enterprise Deployments**: Streamline model distribution in corporate environments
- **Research Projects**: Batch download multiple models for comparative studies

## 🔧 Supported Models

HFdownloader.com supports downloading from the entire Hugging Face Model Hub, including:

- **Language Models**: GPT, BERT, T5, LLaMA, and more
- **Vision Models**: ResNet, Vision Transformers, CLIP, etc.
- **Audio Models**: Whisper, Wav2Vec, speech synthesis models
- **Multimodal Models**: DALL-E, Flamingo, and other cross-modal architectures
- **Fine-tuned Models**: Specialized models for specific tasks and domains

## ⚡ Technical Specifications

- **Web-based Interface**: No downloads or installations required
- **Cross-platform Compatibility**: Works on Windows, macOS, and Linux
- **Modern Browser Support**: Optimized for Chrome, Firefox, Safari, and Edge
- **Efficient Compression**: Smart ZIP packaging for optimal file sizes
- **Robust Download Management**: Handles large repositories and network interruptions

## 🤝 Contributing

We welcome contributions to improve HFdownloader.com! Here's how you can help:

- 🐛 Report bugs and issues
- 💡 Suggest new features and improvements
- 📖 Improve documentation
- 🔧 Submit pull requests

## 📞 Support

- **Website**: [HFdownloader.com](https://hfdownloader.com)
- **Issues**: Report bugs and request features through GitHub Issues
- **Documentation**: Check our documentation for detailed usage guides

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For providing the incredible model hub and ecosystem
- **Open Source Community**: For the tools and libraries that make this project possible
- **Contributors**: Everyone who helps improve this tool

---

<p align="center">
  <strong>Simplifying AI model access, one download at a time.</strong>
</p>

<p align="center">
  Made with ❤️ for the AI community
</p>
