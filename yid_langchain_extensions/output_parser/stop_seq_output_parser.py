from abc import ABC, abstractmethod
from typing import List

from langchain_core.output_parsers import BaseOutputParser


class StopSeqOutputParser(BaseOutputParser, ABC):
    @abstractmethod
    @property
    def stop_sequences(self) -> List[str]:
        pass
