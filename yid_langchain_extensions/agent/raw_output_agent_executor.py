from typing import Optional, Any, Dict

from langchain.agents import AgentExecutor
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.schema import AgentFinish


class RawOutputAgentExecutor(AgentExecutor):
    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        result = super()._return(output, intermediate_steps, run_manager)
        result["raw_output"] = output.log
        return result

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        result = await super()._areturn(output, intermediate_steps, run_manager)
        result["raw_output"] = output.log
        return result
