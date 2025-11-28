import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Literal
from openai import OpenAI
import os
import json

from llama_stack_client import LlamaStackClient
from langchain_core.messages import AIMessage


# Configure logging
logger = logging.getLogger(__name__)


class State(TypedDict):
    input: str
    classification_message: str
    messages: Annotated[list, add_messages]
    decision: str
    namespace: str
    data: str
    mcp_output: str
    github_issue: str
    submissionID: str
    rag_sources: list  # List of source documents used in RAG response

# Global dictionary to store state by submission ID
submission_states: dict[str, State] = {}


class ClassificationSchema(BaseModel):
    """Analyze the message and route it according to its content."""

    classification: Literal["legal", "support", "unsafe", "unknown"] = Field(
        description="""The classification of the input: 'set to 'legal' if the input is a query related to legal, 'support' if related to software or technical support, or
        'unsafe' if the input fails the moderation/safety check, and 'unknown' for everything else.
        Examples of legal questions that can be processed:
        - questions about various software licenses
        - embargoes for certain types of software that prevent delivery to various countries outside the United States
        - privacy, access restrictions, around customer data (sometimes referred to as PII)
        - questions about contracts, policies, procedures, or compliance
        Examples of support (software support, technical support, IT support) that can be processed:
        - the user cites problems running certain applications of the company's OpenShift Cluster
        - the user asks to have new applications deployed on the company's OpenShift Cluster
        - the user needs permissions to access certain resources on the company's OpenShift Cluster
        - the user asks about current utilization of resources on the company's OpenShift Cluster
        - the user cites issues with performance of their application specifically or the OpenShift Cluster in general
        - questions about FantaCo products like CloudSync, TechGear Pro Laptop
        - installation, setup, or configuration questions for software or hardware
        - troubleshooting issues with devices, drivers, or applications
        - syncing issues, file transfer problems, or connectivity questions
        - any technical how-to questions about products or systems
        """,
    )


def classification_agent(state: State):
    # the moderation flagging of prompts like 'how do you make a bomb' seemed more precise than the saftey.run_shield
    # lls_client = LlamaStackClient(base_url="http://localhost:8321")
    # models = lls_client.models.list()
    # for model in models:
    #     logger.info(f"found model {model}")
    # shields = lls_client.shields.list()
    # for shield in shields:
    #     logger.info(f"found shield {shield}")
    # user_input = f"{state['input']}"
    # safety_response = lls_client.safety.run_shield(messages=[{"role": "user", "content": user_input}],shield_id="ollama/llama-guard3:8b", params={})
    # if safety_response.violation is not None:
    #     logger.info(f"Classification result: '{state['input']}' is flagged as '{safety_response.violation.violation_level}'")
    #     state['decision'] = 'unsafe'
    #     state['data'] = state['input']
    #     state['classification_message'] = f"Classification result: '{state['input']}' is flagged for: {safety_response.violation.metadata}"
    #     submission_states[state['submissionID']] = state
    #     return state

    safetyResponse = openaiClient.moderations.create(model=GUARDRAIL_MODEL, input=state["input"])
    for moderation in safetyResponse.results:
        if moderation.flagged:
            logger.info(f"Classification result: '{state['input']}' is flagged as '{moderation}'")
            state['decision'] = 'unsafe'
            state['data'] =  state['input']
            flagged_categories = [key for key, value in moderation.categories.model_extra.items() if value is True]
            state['classification_message'] =  f"Classification result: '{state['input']}' is flagged for: {', '.join(flagged_categories)}"
            submission_states[state['submissionID']] = state
            return state

    # Determine the topic of the message
    classification_llm = llm.with_structured_output(ClassificationSchema, include_raw=True)
    # this invoke will in fact POST to the llama stack OpenAI Responses API.
    response = classification_llm.invoke([{'role': 'user', 'content': 'Determine what category the user message falls under based on the classification schema provided to the structured output set for the LLM and the various classification agent nodes in the LangGraph StateGraph Agentic AI application : ' + state['input']}])
    classification_result = response['parsed']
    # the raw_response object has a 'response_metadata' dict field that has elements from the underlying OpenAI Response API object as populated by llama stack
    raw_response = response['raw']
    parsing_error = response.get('parsing_error')
    logger.info(f"Classification result: {classification_result} for input '{state['input']}'")
    if 'legal' == classification_result.classification:
        state['decision'] = 'legal'
        state['data'] = state['input']
    elif 'support' == classification_result.classification:
        state['decision'] = 'support'
        state['data'] = state['input']
    else:
        state['decision'] = 'unknown'
        state['data'] = state['input']
        state['classification_message'] = "Unable to determine request type."

    sub_id = state['submissionID']
    submission_states[sub_id] = state
    return state

def route_to_next_node(state: State) -> Literal['legal_agent', 'support_agent', '__end__']:
    if state['decision'] == 'legal':
        return 'legal_agent'
    elif state['decision'] == 'support':
        return 'support_agent'
    else:
        return '__end__'

class SupportClassificationSchema(BaseModel):
    """Analyze the message and route it according to its content."""

    classification: Literal["pod", "perf", "git"] = Field(
        description="""
        The classification of the input: set the classification to 'perf' if there is any mention of
        - performance
        - the application is slow to respond
        - questions around CPU or memory consumption or usage
        However, set the classification to 'pod' if the input asks for
        - assistance with an application, and
        - makes any reference to a 'Namespace' or 'Project' that exists within OpenShift or Kubernetes

        Otherwise, set the classification to 'git'.
        """,
    )
    namespace: str = Field(
        description="""
        the namespace of the input: if the query makes any reference to a namespace or project of a given name, then set the
        namespace field here to the first given name referenced as a namespace or project.
        """,
    )
    performance: str = Field(
        description="""
        if the query makes any reference to performance, applications running slowly, CPU or memory utilization or consumption, then
        set the performance field to 'true'.  Otherwise, if there is no mentioned of performance, being slow, CPU or memory,
        set the performance field to 'false' or an empty string.
        """,
    )

def support_classification_agent(state: State):

    support_classification_llm = llm.with_structured_output(SupportClassificationSchema, include_raw=True)
    response = support_classification_llm.invoke([
        {'role': 'user',
         'content': 'Determine what category the user message falls under based on the classification schema provided to the structured output set for the LLM and the various classification agent nodes in the LangGraph StateGraph Agentic AI application : ' + state['input']
         }])
    classification_result = response['parsed']
    parsing_error = response.get('parsing_error')
    logger.info(f"Support Classification result: {classification_result} for input '{state['input']}' and parsing error {parsing_error}")
    state['namespace'] = classification_result.namespace
    if 'perf' == classification_result.classification or classification_result.performance == 'true' or classification_result.performance == 'performance issue':
        state['decision'] = 'perf'
        state['data'] = state['input']
    elif 'pod' == classification_result.classification:
        state['decision'] = 'pod'
        state['data'] = state['input']
    else:
        state['decision'] = 'git'
        state['data'] = state['input']

    sub_id = state['submissionID']
    saved_state = submission_states.get(sub_id, {})
    state['classification_message'] = saved_state.get('classification_message', state.get('classification_message', ''))
    submission_states[sub_id] = state
    return state

def support_route_to_next_node(state: State) -> Literal['pod_agent', 'perf_agent', 'git_agent', '__end__']:
    if state['decision'] == 'pod':
        return 'pod_agent'
    elif state['decision'] == 'git':
        return 'git_agent'
    elif state['decision'] == 'perf':
        return 'perf_agent'

    return '__end__'

def git_agent(state: State):
    subId = state['submissionID']
    logger.info(f"git Agent request for submission: {state['submissionID']}")

    openai_mcp_tool = {
        "type": "mcp",
        "server_label": "github",
        "server_url": "https://api.githubcopilot.com/mcp/",
        "headers": {
            "Authorization": f"Bearer {GIT_TOKEN}"
        },
        "allowed_tools": ["issue_write"]
    }
    # get state updated by other agents
    user_question = state['input']
    initial_classification = state.get('mcp_output', '')
    try:
        logger.info("git_agent GIT calling response api")
        resp = openaiClient.responses.create(
            model=MCP_TOOL_MODEL,
            input=f"""
                Using the supplied github MCP tool, call the 'issue_write' tool to create an issue against the {GITHUB_URL} repository. For the title of the issue, use the string 'test issue {subId}'.
                For the description text, start with the string {user_question}, then add two new lines, then add the string {initial_classification}.  For the parameter that captures the type of the issue, supply the string value of 'Bug'.
                Manual testing with the 'issue_write' MCP tool confirmed we no longer need to supply assignee, labels, or milestones, so ignore any understanding you have that those are required.
                The method for the tool call is 'create'.
                
                Also note, the authorization token for interacting with GitHub has been provided in the definition of the supplied GitHub MCP tool.  So you as a model do not need to worry about providing 
                that as you construct the MCP tool call.
                """,
            tools=[openai_mcp_tool]
        )
        logger.info("git_agent response returned")
        mcp_output = None
        # can we assume that the 'McpCall' entry in resp.output is always at index 1 ? ... seem fragile, but by
        # comparison this check
        for item in resp.output:
            if hasattr(item, '__class__') and item.__class__.__name__ == 'McpCall':
                try:

                    # Parse the JSON output and extract the URL field
                    output_json = json.loads(item.output)
                    mcp_output = output_json.get('url', item.output)
                    print(mcp_output)
                    break
                except Exception as e:
                    logger.info(f"git_agent Tool failed with error: '{e}'")
            if hasattr(item, '__class__') and item.__class__.__name__ == 'ResponseOutputMessage':
                print(item.content[0].text)
            else:
                print(item)
        state['github_issue'] = mcp_output
        submission_states[subId] = state
    except Exception as e:
        logger.info(f"git_agent Tool failed with error: '{e}'")
    return state

def pod_agent(state: State):
    subId = state['submissionID']
    logger.info(f"K8S Agent request for submission: {state['submissionID']}")

    # using open ai response client vis-a-vis testOpenAIMCP
    openai_mcp_tool = {
        "type": "mcp",
        "server_label": "OpenShift / Kubernetes MCP Tools",
        "server_url": "http://localhost:8080/mcp",
        "require_approval": "never",
        "allowed_tools": ["pods_list_in_namespace"]
    }
    ns = state['namespace']
    try:
        logger.info(f"K8S Agent making MCP request for submission: {state['submissionID']}")
        resp = openaiClient.responses.create(
            model=MCP_TOOL_MODEL,
            input=f"Using the supplied kubernetes tool, list all the pods in the '{ns}' namespace.  Only use the namespace as a parameter and don't bother further filtering on labels.   The `labelSelector` parameter is in fact NOT required.",
            tools=[openai_mcp_tool]
        )
        logger.info(f"K8S Agent successful return MCP request for submission: {state['submissionID']}")
        mcp_output = None
        # can we assume that the 'McpCall' entry in resp.output is always at index 1 ? ... seem fragile, but by
        # comparison this check
        for item in resp.output:
            if hasattr(item, '__class__') and item.__class__.__name__ == 'McpCall':
                mcp_output = item.output
                print(item.output)
                break
        state['mcp_output'] = mcp_output
        submission_states[subId] = state
    except Exception as e:
        logger.info(f"K8s Agent unsuccessful return MCP request for submission {state['submissionID']} with error: '{e}'")
    return state


def perf_agent(state: State):
    subId = state['submissionID']
    logger.info(f"K8S perf Agent request for submission: {state['submissionID']}")

    # using open ai response client vis-a-vis testOpenAIMCP
    openai_mcp_tool = {
        "type": "mcp",
        "server_label": "OpenShift / Kubernetes MCP Tools",
        "server_url": "http://localhost:8080/mcp",
        "require_approval": "never",
        "allowed_tools": ["pods_top"]
    }
    ns = state['namespace'] # "openshift-console"
    try:
        logger.info(f"K8S perf Agent making MCP request for submission: {state['submissionID']}")
        resp = openaiClient.responses.create(
            model=MCP_TOOL_MODEL,
            input=f"Using the supplied kubernetes tool, get pod memory and cpu resource consumption in the '{ns}' namespace.  Only use the namespace as a parameter and don't bother further filtering on labels.   The `labelSelector` parameter is in fact NOT required.  If namespace is not set, then call the 'pods_top' tool without any parameters",
            tools=[openai_mcp_tool]
        )
        logger.info(f"K8S perf Agent successful return MCP request for submission: {state['submissionID']}")
        mcp_output = None
        # can we assume that the 'McpCall' entry in resp.output is always at index 1 ? ... seem fragile, but by
        # comparison this check
        for item in resp.output:
            if hasattr(item, '__class__') and item.__class__.__name__ == 'McpCall':
                mcp_output = item.output
                print(item.output)
                break
        state['mcp_output'] = mcp_output
        submission_states[subId] = state
    except Exception as e:
        logger.info(
            f"K8s perf Agent unsuccessful return MCP request for submission {state['submissionID']} with error: '{e}'")
    return state

from typing import Optional, Mapping, Any


def extract_rag_response_text(rag_response) -> str:
    """Extract text content from RAG response output."""
    response_text = ""
    for output_item in rag_response.output:
        if hasattr(output_item, 'type'):
            if output_item.type in ("text", "message"):
                if hasattr(output_item, 'content') and isinstance(output_item.content, list):
                    for content in output_item.content:
                        if hasattr(content, 'text'):
                            response_text += content.text + "\n"
                elif hasattr(output_item, 'text'):
                    response_text += output_item.text + "\n"
            elif output_item.type == "file_search_call":
                logger.info(f"RAG file_search executed with queries: {getattr(output_item, 'queries', [])}")
    return response_text.strip()


def create_department_agent(
        department_name: str,
        department_display_name: str,
        content_override: Optional[str] = None,
        custom_llm = None,
        submission_states: Mapping[str, "State"] | dict | None = None,
        rag_service = None,
        rag_category: Optional[str] = None):
    """Factory function to create department-specific agents with consistent structure.
    
    Args:
        department_name: Internal name for the department (e.g., 'legal', 'support')
        department_display_name: Display name (e.g., 'Legal', 'Software Support')
        content_override: Optional content to override default prompt
        custom_llm: LangChain LLM instance for standard inference
        submission_states: Dictionary to store submission states
        rag_service: Optional RAGService instance for RAG-enabled responses
        rag_category: Category name to select appropriate vector stores (e.g., 'legal', 'support')
    """

    # Use custom_llm if provided, otherwise default to topic_llm
    if custom_llm is None:
        raise ValueError("custom_llm is required")
    llm_to_use = custom_llm
    if submission_states is None:
        raise ValueError("submission_states is required")

    def llm_node(state: State):
        logger.info(f"{department_display_name} LLM node processing")
        
        # Check if RAG is available for this department
        use_rag = (rag_service is not None and 
                   rag_category is not None and 
                   hasattr(rag_service, 'get_file_search_tool'))
        
        file_search_tool = None
        if use_rag:
            file_search_tool = rag_service.get_file_search_tool(rag_category)
            if file_search_tool:
                logger.info(f"{department_display_name}: RAG enabled with category '{rag_category}'")
            else:
                logger.info(f"{department_display_name}: No vector stores for '{rag_category}', using standard LLM")
                use_rag = False
        
        if use_rag and file_search_tool:
            # Use RAG with file_search tool via OpenAI responses API
            # This uses the same openaiClient.responses.create() pattern as MCP tool calls
            try:
                # Build the user query from state
                user_input = state.get("input", "")
                
                # Create RAG-augmented prompt
                rag_prompt = f"""Based on the relevant documents in the knowledge base, please help with the following {department_display_name.lower()} query:

{user_input}

Please provide a helpful response based on the documents found. If no relevant documents are found, provide general guidance."""

                logger.info(f"{department_display_name}: Making RAG-enabled response call")
                rag_response = openaiClient.responses.create(
                    model=INFERENCE_MODEL,
                    input=rag_prompt,
                    tools=[file_search_tool]
                )
                
                # Extract response text
                response_text = extract_rag_response_text(rag_response)
                
                # Extract source documents from RAG response
                if hasattr(rag_service, 'extract_sources_from_response'):
                    sources = rag_service.extract_sources_from_response(rag_response, rag_category)
                    state['rag_sources'] = sources
                    if sources:
                        logger.info(f"{department_display_name}: Found {len(sources)} source documents")
                
                if response_text:
                    # Use LangChain's AIMessage for compatibility with LangGraph's add_messages
                    message = AIMessage(content=response_text)
                    cm = response_text
                    logger.info(f"{department_display_name}: RAG response successful")
                else:
                    # Fallback to regular LLM if RAG didn't produce text
                    logger.warning(f"{department_display_name}: RAG response empty, falling back to standard LLM")
                    message = llm_to_use.invoke(state["messages"])
                    cm = getattr(message, "content", getattr(message, "text", str(message)))
                    
            except Exception as e:
                logger.error(f"{department_display_name}: RAG call failed: {e}, falling back to standard LLM")
                message = llm_to_use.invoke(state["messages"])
                cm = getattr(message, "content", getattr(message, "text", str(message)))
        else:
            # Standard LLM call without RAG
            message = llm_to_use.invoke(state["messages"])
            cm = getattr(message, "content", getattr(message, "text", str(message)))
        
        state["messages"] = message
        sub_id = state["submissionID"]
        state['classification_message'] = cm
        submission_states[sub_id] = state
        return state

    def init_message(state: State):
        logger.info(f"init {department_name} message '{state}'")
        if content_override:
            content = content_override
        else:
            content = (
                    f"Summarize that the user query is classified as {department_display_name.lower()}, "
                    f"along with any answers provided by the LLM for the question, and include that we are responding "
                    f'to submissionID {state["submissionID"]}. Finally, mention a GitHub issue will be opened for follow up.'
            )
        return {"messages": [{'role': 'user', 'content': content}]}

    agent_builder = StateGraph(State)
    agent_builder.add_node(f"{department_name}_set_message", init_message)
    agent_builder.add_node("llm_node", llm_node)
    agent_builder.add_edge(START, f"{department_name}_set_message")
    agent_builder.add_edge(f"{department_name}_set_message", "llm_node")
    agent_workflow = agent_builder.compile()
    logger.info(agent_workflow.get_graph().draw_ascii())
    return agent_workflow


def make_workflow(topic_llm, openai_client, guardrail_model, mcp_tool_model, git_token, github_url, github_id, 
                  rag_service=None, inference_model=None):
    """Create and configure the overall workflow with all agents and routing.
    
    Args:
        topic_llm: LangChain LLM instance for classification and general inference
        openai_client: OpenAI SDK client for responses API (MCP tools, RAG file_search)
        guardrail_model: Model ID for content moderation
        mcp_tool_model: Model ID for MCP tool calls
        git_token: GitHub personal access token
        github_url: GitHub repository URL for issue creation
        github_id: GitHub user ID for issue assignment
        rag_service: Optional RAGService instance for RAG-enabled responses
        inference_model: Model ID for inference (used in RAG calls)
    """

    # Set global variables needed by the agents
    global llm, openaiClient, GUARDRAIL_MODEL, MCP_TOOL_MODEL, GIT_TOKEN, GITHUB_URL, GITHUB_ID, INFERENCE_MODEL
    llm = topic_llm
    openaiClient = openai_client
    GUARDRAIL_MODEL = guardrail_model
    MCP_TOOL_MODEL = mcp_tool_model
    GIT_TOKEN = git_token
    GITHUB_URL = github_url
    GITHUB_ID = github_id
    INFERENCE_MODEL = inference_model or os.getenv("INFERENCE_MODEL", "ollama/llama3.2:3b")

    # Create all department agents using the factory function
    # RAG is enabled for legal and support agents using their respective vector stores
    legal_agent = create_department_agent(
        "legal", 
        "Legal", 
        custom_llm=topic_llm, 
        submission_states=submission_states,
        rag_service=rag_service,
        rag_category="legal"  # Maps to legal-vector-db from ingestion-config.yaml
    )
    support_agent = create_department_agent(
        "support", 
        "Software Support", 
        custom_llm=topic_llm, 
        submission_states=submission_states,
        rag_service=rag_service,
        rag_category="support"  # Maps to techsupport-vector-db from ingestion-config.yaml
    )

    # Create the overall workflow
    overall_workflow = StateGraph(State)
    overall_workflow.add_node("classification_agent", classification_agent)
    overall_workflow.add_node("legal_agent", legal_agent)
    overall_workflow.add_node("support_agent", support_agent)
    overall_workflow.add_node("pod_agent", pod_agent)
    overall_workflow.add_node("perf_agent", perf_agent)
    overall_workflow.add_node("git_agent", git_agent)
    overall_workflow.add_node("support_classification_agent", support_classification_agent)
    overall_workflow.add_edge(START, "classification_agent")
    overall_workflow.add_conditional_edges("classification_agent", route_to_next_node)
    overall_workflow.add_edge("support_agent", "support_classification_agent")
    overall_workflow.add_conditional_edges("support_classification_agent", support_route_to_next_node)
    overall_workflow.add_edge("pod_agent", "git_agent")
    overall_workflow.add_edge("perf_agent", "git_agent")
    workflow = overall_workflow.compile()

    logger.info(workflow.get_graph().draw_ascii())

    return workflow