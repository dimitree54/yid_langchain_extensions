from typing import Optional, Any, Dict, Tuple

from langchain.agents import AgentExecutor
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.schema import AgentFinish, AgentAction


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

    def _get_tool_return(
        self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        result = super()._get_tool_return(next_step_output)
        if isinstance(result, AgentFinish):
            return AgentFinish(
                return_values=result.return_values,
                log=next_step_output[0].log
            )
        return result
