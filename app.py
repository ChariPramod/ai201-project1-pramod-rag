"""Gradio web UI for The Unofficial Guide."""

import gradio as gr

from generator import generate_response


def handle_query(question: str) -> tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "", ""
    result = generate_response(question)
    sources = result["sources"]
    sources_text = "\n".join(f"• {s}" for s in sources) if sources else "(no sources — system refused to answer)"
    return result["answer"], sources_text


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    title="The Unofficial Guide",
) as demo:

    gr.HTML(
        """
        <div style="text-align:center; padding:1.25rem 0 0.5rem;">
            <h1 style="font-size:2rem; font-weight:700; color:#312e81; margin:0;">
                🎓 The Unofficial Guide
            </h1>
            <p style="color:#6b7280; font-size:1rem; margin:0.4rem 0 0;">
                Grounded answers to r/berkeley course &amp; professor questions.
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your question",
                placeholder='e.g. "Is John DeNero a good 61A professor?"',
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer = gr.Textbox(label="Answer", lines=10, interactive=False)
            sources = gr.Textbox(label="Retrieved from", lines=4, interactive=False)

    ask_btn.click(fn=handle_query, inputs=question, outputs=[answer, sources])
    question.submit(fn=handle_query, inputs=question, outputs=[answer, sources])


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  The Unofficial Guide — starting up")
    print("=" * 50 + "\n")
    demo.launch(server_name="127.0.0.1", server_port=7860)
