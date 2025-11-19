from models.model import MyModel
import torch
import torch.nn as nn
# from stock_data_generator_v2 import StockDataGenerator
# from model_trainer import ModelTrainer

from data.data_module import DataModule
from utils.config import Config
from training.trainer import Trainer

cfg = Config("/Users/miloszglowacki/Desktop/code/python/stock_forecast/config/config.yaml")
device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

data_module = DataModule(cfg.data)
train_loader, val_loader, test_loader, scaler = data_module.get_loaders()

model = MyModel(input_size=19, hidden_size=64, num_layers=1)

trainer = Trainer(model, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.001), cfg.training)

history = trainer.train(train_loader, val_loader)

print(history)

# trainer = ModelTrainer(
#     model=model,
#     loss_fn=nn.MSELoss(),
#     optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
#     device="cuda" if torch.cuda.is_available() else "cpu",
# )

# train_history = trainer.fit(
#     train_loader=train_loader,
#     val_loader=val_loader,
#     epochs=10,
# )

# eval_history = trainer.evaluate(test_loader)

# trainer.plot_history()

# trainer.plot_predictions(test_loader)

# print(train_history)
# print(eval_history)