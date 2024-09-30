class ModelConfiguration:

    def __int__(self,
                vocab_size=50257,
                context_length=1024,
                embedding_dimension=768,
                num_attention_heads=12,
                num_layers=12,
                dropout_rate=0.0,
                feed_forward_output_scale=4,
                qkv_bias=False):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dimension = embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.qkv_bias = qkv_bias
