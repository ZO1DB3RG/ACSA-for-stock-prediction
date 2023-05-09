### Use following codes to construct the BART generation object and train it:
```py
model = AspectAnything(
            encoder_decoder_type="bart",
            encoder_decoder_name="facebook/bart-base",
            args=model_args,
        )
best_accuracy = model.train_model(train_df, best_accuracy)
```
