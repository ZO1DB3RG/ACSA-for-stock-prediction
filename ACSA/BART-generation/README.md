In modeling_bert.py we change the loss function
```py
labels = labels.to(lm_logits.device)
loss_fct = SelfAdjDiceLoss(reduction='sum', alpha=0.4)
# loss_fct = CrossEntropyLoss()
masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
```
