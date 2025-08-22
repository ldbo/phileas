# Logging

Phileas uses the `phileas` logger. Additionally, each bench instrument `ins` has
its own logger named `phileas.ins`. However, by default you won't see many
logs. The simplest way to turn them on is to run:

```python
import logging

logging.basicConfig(level=logging.INFO)
```
