from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from semantic_image_search.backend.config import Config
from semantic_image_search.backend.logger import GLOBAL_LOGGER as log
from semantic_image_search.backend.exception.custom_exception import SemanticImageSearchException


class QueryTranslator:
    """
    LLM-based Query Rewriter for CLIP-style image caption search.
    """

    def __init__(self):
        try:
            log.info("Initializing QueryTranslator...", model=Config.OPENAI_MODEL)

            self.llm = ChatOpenAI(
                model=Config.OPENAI_MODEL,
                temperature=0,
                timeout=20,     # prevents API hangs
            )

            self.prompt_template = PromptTemplate(
                input_variables=["input_query"],
                template="""
You are an expert at rewriting queries for the CLIP image–text model.

Goal:
Rewrite the user query into a short, concrete, descriptive image caption.
The rewritten query must maximize CLIP retrieval accuracy.

Guidelines:
- Keep the original meaning.
- Use 3–12 word caption style.
- Remove chat words (show me, give me, please, etc.)
- Keep colors, objects, actions.
- Translate to English if needed.
- Do NOT add new details.

User Query: {input_query}

Respond with only the rewritten caption.
                """.strip(),
            )

            log.info("QueryTranslator initialized successfully")

        except Exception as e:
            log.error("Failed to initialize QueryTranslator", error=str(e))
            raise SemanticImageSearchException("Failed to initialize QueryTranslator", e)

    def translate(self, user_query: str) -> str:
        """Run LLM and translate chat input → caption."""
        if not isinstance(user_query, str) or not user_query.strip():
            log.error("Invalid input query for translation", query=user_query)
            raise ValueError("Query must be a non-empty string")

        log.info("Translating query", input_query=user_query)

        try:
            prompt = self.prompt_template.format(input_query=user_query)
            log.info("Sending translation prompt to LLM")

            # SAFER & RECOMMENDED METHOD
            final_caption = self.llm.invoke(prompt).content.strip()

            log.info(
                "Translation completed",
                original=user_query,
                translated=final_caption
            )
            return final_caption

        except Exception as e:
            log.error("LLM translation failed", query=user_query, error=str(e))
            raise SemanticImageSearchException("LLM translation failed", e)


# ---- Lazy Singleton ----
_translator_instance = None

def translate_query(user_query: str) -> str:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = QueryTranslator()
    return _translator_instance.translate(user_query)
