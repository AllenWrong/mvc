from pydantic import BaseModel

# some module use config.DATA_DIR, so here we import the variable from constants module
from src.config.constants import *


class Config(BaseModel):
    @property
    def class_name(self):
        return self.__class__.__name__
