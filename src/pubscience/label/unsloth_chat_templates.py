from unsloth.chat_templates import CHAT_TEMPLATES
import argparse


def get_chat_template(template_name: str):
    """
    Get the chat template by name.
    :param template_name: Name of the chat template.
    :return: The chat template.
    """

    if template_name not in CHAT_TEMPLATES:
        raise ValueError(f"Chat template '{template_name}' not found. Available templates: {list(CHAT_TEMPLATES.keys())}")

    return CHAT_TEMPLATES[template_name]

def main():
    parser = argparse.ArgumentParser(description="Get chat template by name.")
    parser.add_argument("--template_name", type=str, help="Name of the chat template to retrieve.")
    args = parser.parse_args()

    print("Loading chat template options..")
    print("Available chat templates:", CHAT_TEMPLATES.keys())

    try:
        template = get_chat_template(args.template_name)
        print(f"Chat template '{args.template_name}' retrieved successfully.")
        print("Template content:", template)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()