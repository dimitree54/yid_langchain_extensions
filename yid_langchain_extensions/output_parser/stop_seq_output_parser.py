from abc import ABC
from typing import List, Optional

from langchain_core.output_parsers import BaseOutputParser


class StopSeqOutputParser(BaseOutputParser, ABC):
    @property
    def stop_sequences(self) -> Optional[List[str]]:
        return None
