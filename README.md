# AI-Runtime-Recommendation-Deployment-System
本项目实现本地大模型运行时智能调度系统，解决开发者本地部署AI时的模型选择、硬件适配与多后端管理问题。支持Ollama与LM Studio双后端统一调度，自动检测GPU、Apple Silicon及CPU环境，基于显存、内存、磁盘空间进行模型兼容性分析与推荐。内置多维模型评分机制，结合HumanEval、MMLU、上下文长度、推理速度、中文能力等指标，针对编程、写作、推理、对话等场景动态加权排序。支持benchmark与真实token/s性能测试，持续优化推荐结果。提供Web UI、CLI、PyQt GUI三端交互，实现模型自动拉取、运行与生命周期管理。已完成本地AI环境初始化、大模型自动部署、多Runtime统一调度及低配置设备适配。
