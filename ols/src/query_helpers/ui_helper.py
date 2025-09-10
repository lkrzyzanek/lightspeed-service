import json
import logging
from langchain_core.messages import ToolMessage
from next_gen_ui_agent import InputData, NextGenUIAgent
from next_gen_ui_agent.model import LangChainModelInference
from next_gen_ui_agent.types import Rendition
import yaml

logger = logging.getLogger(__name__)


def create_ui_agent(llm) -> NextGenUIAgent:
    inference = LangChainModelInference(llm)
    return NextGenUIAgent({"inference": inference})


async def generate_ui(
    next_gen_ui_agent: NextGenUIAgent,
    user_prompt: str,
    tool_calls_messages: list[ToolMessage],
) -> list[Rendition]:

    # TODO: Move that 4 steps logic to NGUI agent for easier consumtion

    # Step 1: data selection
    input_data: list[InputData] = []
    for m in tool_calls_messages:
        # logger.debug(m.content)
        if m.type == "tool" and (m.status and m.status == "success"):
            input: InputData = {
                "id": m.tool_call_id,
                "data": m.content,
                "type": m.name,
            }
            # TODO: Better handle non JSON content like YAML from MCP
            # TODO: Remove pyaml dependency package
            if not m.content.startswith("{"):
                # convert openshift MCP from yaml to json
                data = yaml.load(m.content, Loader=yaml.CLoader)
                data_wrapped = {m.name: data}
                logger.info(
                    json.dumps(data_wrapped, ensure_ascii=False, default=str, indent=2)
                )
                input["data"] = json.dumps(data_wrapped, default=str)

            input_data.append(input)

    # Step 2: component selection
    components = await next_gen_ui_agent.component_selection(
        {"user_prompt": user_prompt, "input_data": input_data}
    )
    # Step 3: transformation
    components_transformed = next_gen_ui_agent.data_transformation(
        input_data=input_data, components=components
    )
    # Step 4: rendition
    renditions = next_gen_ui_agent.design_system_handler(
        components=components_transformed, component_system="json"
    )
    return renditions
