[LLM-REDIAL: A Large-Scale Dataset for Conversational Recommender
Systems Created from User Behaviors with LLMs](https://aclanthology.org/2024.findings-acl.529.pdf)

Deep Cleaning and Demoralizing LMSYS's 1m Chat Dataset
Resources
Recently, I decided to do some data analysis on the LMSYS-chat-1m dataset—probably the most diverse collection of human-generated prompts we have.

Then DeepSeek came along, so I re-prompted the entire cleaned dataset. I also used a classifier to flag moralizing entries (hard refusals, soft refusals, annoying warnings and important notes, etc.) Overall, DeepSeek is probably the least censored model out of all corporate models I've tested with the only other contender being Mistral Large.

There's a bunch of other goodies I included as well.

Here it is: https://huggingface.co/datasets/OpenLeecher/lmsys_chat_1m_clean


[LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset](https://arxiv.org/abs/2309.11998)