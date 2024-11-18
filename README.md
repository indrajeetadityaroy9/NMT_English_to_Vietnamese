# Neural Machine Translation (NMT) Between English and Vietnamese

Pipeline for sequence-to-sequence translation using encoder-decoder RNN architecture with attention mechanism. Designed for efficient training and inference, the pipeline is tailored for the IWSLT 2015 English-Vietnamese dataset, incorporating quantization for optimized performance. The pipeline includes:

- **Data Handling**: Automated loading, preprocessing, tokenization, and vocabulary creation for English and Vietnamese text using NLTK and Pyvi. Preprocessing also includes handling punctuation, contractions, and HTML entities for clean input data.

- **Dynamic Model Architecture**: Custom encoder-decoder models with bidirectional GRU layers and attention mechanisms for context-aware translation. The architecture supports dynamic adjustments to vocabulary size, embedding dimensions, and hidden layer configurations.

- **Training Workflow**: Features gradient accumulation, mixed precision training with torch.cuda.amp, learning rate scheduling via StepLR, and model checkpointing based on BLEU score evaluation.

- **Evaluation and Inference**: Tools for BLEU-based evaluation, including sentence-level and corpus-level scores, and support for dynamic quantization of trained models to reduce size and improve inference speed.

- **Quantization**: Applied to encoder and decoder models to achieve significant size reduction and faster inference while maintaining translation quality.

- **Pretrained Integration**: Utilizes the facebook/nllb-200-3.3B model for dataset preprocessing, enabling accurate and efficient language-specific translations.

- **CLI**: Flexible options for training, validation, and testing, with support for batch processing and deployment-ready quantized model inference.

### **Translation Metrics**

 - tst2012.txt - **Average Sentence BLEU:** 42.198
 - tst2012.txt - **Corpus BLEU:** 46.388
 
 - tst2013.txt - **Average Sentence BLEU:** 41.510
 - tst2013.txt - **Corpus BLEU:** 45.493
