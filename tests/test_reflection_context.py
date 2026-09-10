"""Check lossless references and counted repetition in reflection context."""

from copy import deepcopy

from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.reflection_context import compact_reflection_records


def test_exact_duplicates_reference_the_first_occurrence_without_mutating_records() -> None:
    """Keep distinct evidence and example multiplicity while referencing large repeats."""
    passage = "A substantial retrieved passage with supporting facts. " * 20
    feedback = "A shared explanation of how this component works. " * 10
    records = [
        {"Inputs": {"question": "First?", "passages": [passage, passage]}, "Feedback": feedback + "\n\nFIRST_ERROR"},
        {"Inputs": {"question": "Second?", "passages": [passage]}, "Feedback": feedback + "\n\nSECOND_ERROR"},
    ]
    original = deepcopy(records)
    compact = compact_reflection_records(records)
    assert records == original
    assert compact[0]["Inputs"]["passages"][0] == passage
    assert compact[0]["Inputs"]["passages"][1] == "[Repeated text; see Example 1 / Inputs / passages / Item 1.]"
    assert passage not in str(compact[1])
    assert "Example 1 / Feedback / paragraph 1" in compact[1]["Feedback"]
    assert "FIRST_ERROR" in str(compact[0]) and "SECOND_ERROR" in str(compact[1])
    assert len(compact) == len(records)


def test_long_unique_evidence_survives_and_repeated_lines_keep_their_count() -> None:
    """Retain late evidence past 8,000 characters while collapsing log spam."""
    evidence = "\n".join(f"Step {index}: distinct command and result {index}" for index in range(500))
    repeated = "still waiting for the package download\n" * 500
    compact = compact_reflection_records([{"Output": repeated + "FINAL_ERROR\n" + evidence}])
    assert evidence in compact[0]["Output"]
    assert "FINAL_ERROR" in compact[0]["Output"]
    assert "repeated 499 additional times" in compact[0]["Output"]
    assert compact[0]["Output"].count("still waiting") == 1


def test_stateless_prompt_does_not_repeat_the_current_document() -> None:
    """Reference the already rendered instruction instead of copying it into every example."""
    instruction = "Use this detailed current instruction when answering. " * 20
    prompt = InstructionProposalSignature.prompt_renderer(
        {
            "current_instruction_doc": instruction,
            "dataset_with_feedback": [{"Document": instruction, "Feedback": "fail"}],
        }
    )
    assert isinstance(prompt, str)
    assert prompt.count(instruction) == 1
    assert "see Current instruction above" in prompt


def test_reference_state_is_local_to_each_prompt() -> None:
    """Never refer to evidence from another component or previous request."""
    text = "This unique evidence must remain in both independent prompts. " * 20
    records = [{"Inputs": text}]
    assert compact_reflection_records(records) == records
    assert compact_reflection_records(records) == records
