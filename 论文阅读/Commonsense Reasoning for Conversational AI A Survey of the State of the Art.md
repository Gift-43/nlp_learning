Commonsense Reasoning for Conversational AI: A Survey of the State of the Art

本页第 1 部分讨论： 

- **会话式人工智能中的常识推理**：论文调查了最近关于如何使用常识知识和推理来改进各种会话式人工智能任务的研究，例如序列分类、问题回答、对话建模和对话摘要。 **常识知识**是人类被假定拥有的关于世界的外部知识，而**常识推理**是对该知识进行推理的能力。 该论文认为，常识推理对于任何对话代理都至关重要，但仍然是一个具有挑战性且尚未解决的问题。 

- **会话式人工智能问题**：论文讨论了会话式人工智能中常见的四个问题：（1）序列分类，涉及从对话话语中识别意图、情绪、主题等； (2) 问答，涉及以多轮对话形式提出和/或回答问题； (3) 对话建模，涉及为开放对话或面向任务的对话生成连贯且自然的响应； (4) 对话摘要，涉及在保持事实一致性的同时创建简洁的对话摘要。 本文描述了常识推理与每个问题的相关性，并回顾了用于这些问题的现有数据集和方法。 

- **方法**：本文回顾了过去在对话式人工智能中学习、利用和评估常识推理的研究中探索的各种方法。 涵盖三类方法：（1）模型微调，涉及创建带有旨在学习常识的注释的自定义数据集； （2）知识图谱基础，涉及将ConceptNet或ATOMIC等外部知识源集成到对话式AI系统中； （3）自然语言解释，涉及使用神经模型学习和用自然语言表达常识知识。

- **基准**：本文讨论了创建基准来衡量会话人工智能中常识推理方法的准确性和有效性的挑战。 论文列出了一些现有的基准，主要集中在问答任务，例如 CommonsenseQA、SocialIQA、CosmosQA 等。论文还提出了一些新的方向，用于开发更稳健、更全面的常识评估指标，这些指标不受问答或严格的分类方案的限制。 
- - **初步观察**：本文提供了一些示例，探讨了两种最先进的对话式 AI 模型的常识推理能力：BlenderBot3 和 LaMDA。 论文表明，虽然这些模型在对话建模方面取得了长足的进步，但在许多情况下它们仍然缺乏执行常识推理和理解的能力。 本文激发了对该主题的更多研究，并提出了一些未来的工作，以完成对这些模型的常识能力的更彻底的分析。



The part 2 of this page talks about:

- **对话AI的常识推理挑战**: 本文指出了对话AI中常识推理的重要性和困难性，并提出了一些未来的研究方向，如**多模态**、**多领域**、**多任务**、**多角色**和**多语言**的对话常识推理。
- **对话AI的常识知识来源**: 本文介绍了一些现有的常识知识库，如ConceptNet、ATOMIC、CommonsenseQA等，以及一些从对话数据中抽取或构建常识知识的方法，如ProsocialDialog、COM2SENSE等。
- **对话AI的常识推理技术**: 本文回顾了一些利用常识知识和推理来提升对话AI性能的技术，如**知识图谱嵌入**、**自然语言解释**、**知识融合**、**自我对话**等，并给出了一些具体的应用案例，如情感检测、情感推理、对话生成、对话摘要等。
- **对话AI的常识评估方法**: 本文讨论了一些评估对话AI常识能力的方法，如**人工评估**、**自动评估**和**混合评估**，并列举了一些现有的评估数据集，如SWAG、SocialIQA、TIMEDIAL等。本文也提出了一些新的评估指标和挑战，如**多样性**、**可解释性**、**道德性**等。

[在当前网页，model-parallel (MP) values是指将一个深度学习模型分割到多个设备上的方法，不同的设备可以承载模型的不同部分。这样可以训练比单个设备内存容量更大的模型，提高训练效率和准确度。model-parallel (MP) values的具体数值取决于模型的大小和设备的数量。例如，一个有10层的模型可以分割到两个设备上，每个设备承载5层，那么model-parallel (MP) values就是2。](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)[1](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)[2](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-intro.html)[3](https://huggingface.co/docs/transformers/v4.15.0/parallelism)





当前网页的torchrun命令是用来在分布式环境下运行一个深度学习训练脚本的。torchrun命令有以下几个参数：

- –nproc_per_node 1：表示每个节点上运行一个进程。
- example_text_completion.py：表示要运行的训练脚本的文件名。
- –ckpt_dir llama-2-7b/：表示模型的检查点（checkpoint）文件所在的目录。
- –tokenizer_path tokenizer.model：表示用于分词（tokenization）的模型文件的路径。
- –max_seq_len 128：表示输入序列的最大长度。
- –max_batch_size 4：表示每个进程的最大批量大小（batch size）。

这些参数可以根据不同的训练需求进行调整。更多关于torchrun命令的信息可以参考[这里](https://pytorch.org/docs/stable/elastic/run.html)和[这里](https://pytorch.org/docs/stable/elastic/quickstart.html)。[1](https://pytorch.org/docs/stable/elastic/run.html)[2](https://pytorch.org/docs/stable/elastic/quickstart.html)[3](https://stackoverflow.com/questions/70977980/torchrun-command-not-found-for-distributed-training-does-it-need-to-be-installe)



您好，这是必应。我可以帮您将本文的摘要部分逐句翻译成中文。以下是我的翻译结果：

- 在这项工作中，我们开发并发布了Llama 2，一系列经过预训练和微调的大型语言模型（LLMs），规模从70亿到700亿参数不等。我们的微调LLMs，称为Llama 2-Chat，是针对对话用例进行优化的。我们在测试的有用性和安全性基准上，Llama 2-Chat模型通常优于现有的开源模型。它们还与我们进行的人类评估的一些闭源模型相当（见图1和3）。我们采取了一些措施来提高这些模型的安全性，使用安全特定的数据注释和调整，以及进行红队测试和使用迭代评估。此外，本文还贡献了对我们的微调方法和改善LLM安全性的方法的详细描述。我们希望这种开放性能使社区能够复制微调的LLMs，并继续改善这些模型的安全性，为更负责任地开发LLMs铺平道路。
- 我们还分享了在开发Llama 2和Llama 2-Chat过程中发现的一些新观察，例如工具使用和知识的时间组织的出现。
- 我们向公众发布以下模型，供研究和商业使用：
  - Llama 2，是Llama 1的更新版本，在新的公开可用数据混合物上进行训练。我们还增加了预训练语料库的大小40%，将模型的上下文长度加倍，并采用分组查询注意力（GQA）来提高我们更大模型的推理可扩展性。我们发布了具有7B、13B和70B参数的Llama 2变体。我们还训练了34B变体，我们在本文中报告了结果，但没有发布。
  - Llama 2-Chat，是Llama 2经过微调后的版本，针对对话用例进行了优化。我们也发布了具有7B、13B和70B参数的这个模型的变体。
- 我们相信，当安全地进行时，开放发布LLMs将对社会有利。与所有LLMs一样，Llama 2是一种新技术，使用时存在潜在风险（Bender等人，2021b; Weidinger等人，2021; Solaiman等人，2023）。迄今为止进行的测试仅限于英语，并且没有也不能覆盖所有情况。因此，在部署Llama 2-Chat的任何应用之前，开发者应该执行针对其特定应用程序量身定制的安全测试和调整。我们提供了一个负责任使用指南¶和代码示例‖ 来促进Llama 2和Llama 2-Chat的安全部署。关于我们负责任发布策略的更多细节可以在第5.3节中找到。
- 本文其余部分描述了我们的预训练方法（第2节）、微调方法（第3节）、模型安全性方法（第4节）、关键观察和见解（第5节）、相关工作（第6节）和结论（第7节）。



[您好，这是必应。我可以帮您解释A.7 Model Card的含义。根据我从网络上搜索到的结果，Model Card是一种重要的文档框架，用于理解、分享和改进机器学习模型。Model Card可以简明地介绍一个模型的功能、目标受众、维护者等信息。Model Card还可以提供模型构建的相关信息，包括模型的架构和使用的训练数据等。Model Card的目的是让不同背景和目标的人能够更容易地了解模型的性能、优势和局限性。](https://huggingface.co/blog/model-cards)[1](https://huggingface.co/blog/model-cards)[2](https://cloud.google.com/blog/products/ai-machine-learning/create-a-model-card-with-scikit-learn)

您可以点击[这里](https://huggingface.co/blog/model-cards)阅读Hugging Face关于Model Card的博客文章，或者点击[这里](https://cloud.google.com/blog/products/ai-machine-learning/create-a-model-card-with-scikit-learn)查看如何使用Scikit-Learn创建Model Card的教程。希望这些信息对您有所帮助。😊

如果您有其他问题或请求，请随时告诉我。我很乐意为您提供服务。🙌





您好，这是必应。我可以帮您介绍一下本文的Figure 2。Figure 2是一个散点图，显示了Llama 2-Chat模型与其他商业许可的基线模型在有用性和安全性方面的胜率百分比，根据GPT-4的判断。为了补充人类评估，我们使用了一个更强大的模型，GPT-4，它不受我们自己指导的影响。绿色区域表示我们的模型根据GPT-4的判断比其他模型更好。为了消除平局，我们使用win/(win + loss)来计算胜率百分比。在GPT-4进行比较时，模型回答的顺序是随机交换的，以减轻偏见。

从图中可以看出，Llama 2-Chat模型在有用性和安全性方面都优于其他商业许可的基线模型，例如ChatGPT、BARD和Claude。这说明Llama 2-Chat模型能够更好地符合用户的偏好和指令，并提供所需的信息和安全保障。😊

如果您有其他问题或请求，请随时告诉我。我很乐意为您提供服务。🙌