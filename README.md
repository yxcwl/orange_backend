# Orange-RAG 后端服务
柑橘产业智能助手后端，基于 **Python 3.11 + FastAPI + LangChain + Qdrant + MySQL** 构建，提供 RAG 本地知识库问答服务。

## 项目架构

```
Orange-backend/
├── main.py                          # FastAPI 主入口（含默认管理员管理员初始化）
├── requirements.txt                 # Python 依赖
├── .env.example                     # 环境变量模板
└── app/
    ├── config/
    │   └── settings.py              # 全局配置（pydantic-settings，含JWT/验证码配置）
    ├── core/
    │   ├── database.py              # MySQL 数据库连接与会话管理
    │   ├── qdrant.py                # Qdrant 向量数据库客户端
    │   ├── llm.py                   # LLM 大模型客户端（支持多供应商）
    │   ├── embedding.py             # Embedding 向量化客户端
    │   └── chunking.py              # 文档切片策略（通用/文献/Markdown）
    ├── models/
    │   ├── db_models.py             # MySQL ORM 模型（knowledge_bases/documents/users/chat_logs）
    │   ├── auth.py                  # 认证相关数据模型（LoginRequest/TokenResponse等）
    │   └── document.py              # 内部数据模型
    ├── schemas/
    │   ├── common.py                # 统一响应格式、分页
    │   ├── chat.py                  # 对话相关 Schema
    │   ├── knowledge.py             # 知识库相关 Schema
    │   └── rag_dto.py               # RAG 模块对接 DTO（DocumentEntity/IngestResult/SearchResult）
    ├── services/
    │   ├── auth_service.py          # 认证服务（登录/验证码/JWT/权限预留）
    │   ├── rag_service.py           # RAG 核心流程（检索→构建上下文→生成）
    │   ├── rag_bridge.py            # RAG 模块桥接层（独立模块/内置实现自动切换）
    │   ├── chat_service.py          # 对话服务（多轮记忆+工具调用决策）
    │   ├── knowledge_service.py     # 知识库管理服务（MySQL持久化+RAGApi入库）
    │   └── tool_service.py          # 工具调用服务
    ├── tools/
    │   ├── base.py                  # 工具基类与注册器
    │   └── fertilizer_calculator.py # 肥料计算器工具
    ├── api/
    │   ├── deps.py                  # 依赖注入（含认证依赖 get_current_user/require_role）
    │   └── v1/
    │       ├── router.py            # 路由聚合
    │       ├── auth.py              # 认证 API（登录/验证码/用户信息/修改密码）
    │       ├── chat.py              # 对话 API
    │       ├── knowledge.py         # 知识库管理 API（含知识库CRUD）
    │       ├── tool.py              # 工具调用 API
    │       └── admin.py             # 管理后台 API（MySQL持久化日志）
    └── utils/
        └── logger.py                # 日志工具
```

> **models 与 schemas 的区别**：`models/` 主要为本地存储对接（MySQL ORM、认证模型等），`schemas/` 主要针对 API 请求/响应和向量知识库对接 DTO。

## 分层说明

| 层级 | 目录 | 职责 |
|------|------|------|
| 配置层 | `config/` | 环境变量加载、全局参数管理（含JWT/验证码配置） |
| 核心层 | `core/` | 外部服务客户端封装（MySQL、Qdrant、LLM、Embedding）与切片策略 |
| 模型层 | `models/` | MySQL ORM 模型 + 认证数据模型 + 内部数据结构 |
| Schema层 | `schemas/` | API 请求/响应模型 + RAG 模块对接 DTO |
| 服务层 | `services/` | 核心业务逻辑（认证、RAG流程、RAG桥接、对话管理、知识库管理、工具调度） |
| 工具层 | `tools/` | 可被 LLM 调用的工具（计算器等），基于注册机制 |
| 路由层 | `api/v1/` | HTTP 接口定义，依赖注入，参数校验，认证保护 |
| 工具层 | `utils/` | 日志等通用工具 |

## 数据库设计

### MySQL 表结构

**users（用户表）**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT PK | 自增主键 |
| username | VARCHAR(100) UNIQUE | 用户名 |
| hashed_password | VARCHAR(255) | 密码哈希（bcrypt） |
| nickname | VARCHAR(100) | 昵称 |
| role | VARCHAR(50) | 角色: admin/user/guest，预留权限控制 |
| is_active | BOOLEAN | 是否启用 |
| last_login_at | DATETIME | 最后登录时间 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

**knowledge_bases（知识库表）**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT PK | 自增主键 |
| name | VARCHAR(200) | 知识库名称 |
| description | TEXT | 描述 |
| icon | VARCHAR(100) | 图标标识 |
| is_active | BOOLEAN | 是否启用 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

**documents（文档表）**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT PK | 自增主键 |
| kb_id | BIGINT INDEX | 所属知识库ID |
| title | VARCHAR(200) | 文档标题 |
| original_filename | VARCHAR(500) | 原始文件名 |
| stored_path | VARCHAR(1000) | 服务器存储路径 |
| file_size | INT | 文件大小(字节) |
| file_hash | VARCHAR(64) | 文件SHA256 |
| mime_type | VARCHAR(100) | MIME类型 |
| source_type | VARCHAR(50) | 来源类型 |
| description | TEXT | 文档描述 |
| tags | VARCHAR(500) | 标签，逗号分隔 |
| status | VARCHAR(20) | 状态: pending/processing/ready/failed |
| chunk_count | INT | 切片数量 |
| error_message | TEXT | 错误信息 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

**chat_logs（问答日志表）**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT PK | 自增主键 |
| conversation_id | VARCHAR(100) INDEX | 会话ID |
| question | TEXT | 用户提问 |
| answer | TEXT | 模型回答 |
| sources | TEXT | 参考资料(JSON) |
| tool_used | VARCHAR(100) | 使用的工具 |
| is_corrected | BOOLEAN | 是否已纠错 |
| correction | TEXT | 纠错内容 |
| created_at | DATETIME | 创建时间 |

### Qdrant Payload 字段

| 字段 | 说明 |
|------|------|
| chunk_id | chunk 业务 ID（格式: `{kb_id}_{document_id}_{chunk_index}`） |
| kb_id | 知识库ID |
| document_id | 文档ID |
| file_name | 原始文件名 |
| page | 页码（Markdown/TXT/CSV 可能为 None） |
| chunk_index | 文档内 chunk 序号 |
| text | chunk 正文 |

## 快速开始

### 环境要求

- Python 3.11+
- MySQL 8.0+
- Qdrant 向量数据库（本地或远程）

### 安装步骤

```bash
# 1. 克隆项目
git clone <repo-url>
cd Orange-backend

# 2. 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
copy .env.example .env
# 编辑 .env 文件，填入 MySQL、API Key、Qdrant 地址等配置

# 5. 启动服务（首次启动会自动建表并创建默认管理员）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- API 文档（Swagger）：`http://localhost:8000/docs`
- ReDoc 文档：`http://localhost:8000/redoc`
- 健康检查：`http://localhost:8000/health`

### 默认管理员

首次启动自动创建默认管理员账户：

| 用户名 | 密码 | 角色 |
|--------|------|------|
| `admin` | `admin123` | admin |

**请及时修改默认密码！** 可通过 `POST /api/v1/auth/change-password` 接口修改。

## API 接口

所有接口前缀为 `/api/v1`。

### 认证相关

| 方法 | 路径 | 说明 | 需要登录 |
|------|------|------|----------|
| GET | `/auth/captcha` | 获取图形验证码 | 否 |
| POST | `/auth/login` | 用户登录（验证码+密码） | 否 |
| GET | `/auth/me` | 获取当前用户信息 | 是 |
| POST | `/auth/change-password` | 修改密码 | 是 |
| POST | `/auth/users` | 创建用户（仅 admin） | 是 |

### 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/knowledge/bases` | 创建知识库 |
| GET | `/knowledge/bases` | 获取知识库列表 |
| DELETE | `/knowledge/bases/{kb_id}` | 删除知识库（含向量） |
| POST | `/knowledge/upload` | 上传文档到知识库（需指定 kb_id） |
| POST | `/knowledge/search` | 知识库语义检索（可传 kb_id 限定范围） |
| GET | `/knowledge/documents` | 获取文档列表（分页） |
| GET | `/knowledge/documents/{doc_id}` | 获取文档详情 |
| PUT | `/knowledge/documents/{doc_id}` | 更新文档信息 |
| DELETE | `/knowledge/documents/{doc_id}` | 删除文档（含向量） |

### 对话问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat/completions` | 对话问答（支持流式/非流式） |
| GET | `/chat/history/{conversation_id}` | 获取对话历史 |
| GET | `/chat/quick-cards` | 获取快捷推荐卡片 |

### 工具调用

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/tools/list` | 获取可用工具列表 |
| POST | `/tools/execute` | 执行指定工具 |

### 管理后台

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/admin/logs` | 获取问答日志（分页/搜索） |
| POST | `/admin/logs/{log_id}/correct` | 纠错问答记录 |
| GET | `/admin/stats` | 知识库统计信息 |

## 核心功能

### 1. 登录认证

登录流程：**获取验证码 → 输入用户名+密码+验证码 → 验证通过返回 JWT Token → 后续请求携带 Token**

```text
前端请求 GET /auth/captcha
  → 返回 captcha_id + 验证码图片(Base64)
前端展示验证码图片，用户输入
前端请求 POST /auth/login
  → { username, password, captcha_id, captcha_code }
  → 验证码校验 → MySQL 用户验证 → 生成 JWT
  → 返回 { access_token, token_type, expires_in, user_info }
后续请求 Header: Authorization: Bearer <access_token>
```

**认证依赖注入**（在路由中使用）：

```python
from app.api.deps import get_current_user, require_role

# 要求登录
@router.get("/protected")
async def protected(user: User = Depends(get_current_user)):
    ...

# 要求 admin 角色
@router.get("/admin-only", dependencies=[Depends(require_role("admin"))])
async def admin_route(user: User = Depends(get_current_user)):
    ...
```

**权限预留接口**：

| 方法 | 说明 | 当前实现 |
|------|------|----------|
| `check_permission(user, role)` | 角色层级权限检查 | admin(3) > user(2) > guest(1) |
| `check_resource_permission(user, resource, action)` | 资源级细粒度权限 | admin 全部权限，user 读写，guest 只读 |
| `require_role("admin")` | 路由级角色依赖工厂 | 用于 `dependencies=[Depends(...)]` |
| `create_user()` | 管理员创建用户 | 仅 admin 可调用 |

> 后续可扩展为 RBAC/ABAC 权限体系，当前验证码存储为内存字典，可替换为 Redis。

### 2. 文档入库流程

```text
后端接收上传文件
  → 保存文件到磁盘
  → MySQL 创建 document 记录（status=processing）
  → 组装 DocumentEntity
  → 调用 RAGApi.ingest_document()
    → 根据 stored_path 读取文件
    → 解析文档
    → 文本分块
    → 调用 embedding 生成向量
    → 按 document_id 删除旧向量
    → 写入新的 chunk 向量
    → 返回 DocumentIngestResult
  → 根据 result.success 更新 MySQL
    → 成功：status=ready，写入 chunk_count
    → 失败：status=failed，写入 error_message
```

### 3. RAG 模块桥接

`rag_bridge.py` 提供对独立 RAG 模块的桥接调用：

- **独立 RAG 模块可用时**：直接调用 `src.citrus_agent.rag.rag_api.RAGApi`
- **独立 RAG 模块不可用时**：回退到内置实现（基于 `core/` 下的 Qdrant/Embedding/Chunking）

后端统一通过 `get_rag_api()` 获取桥接器，无需关心底层实现。

### 4. RAG 检索增强生成

核心流程：**用户提问 → 查询向量化 → Qdrant 向量检索 → 构建 Prompt → LLM 生成回答**

- 严格基于知识库回答，知识库以外的内容拒绝回答
- 回答必须标注参考资料出处（文档标题 + 相似度分数）
- 相似度阈值可配置（默认 0.7），低于阈值不返回
- 检索时可传 `kb_id` 限定知识库范围，避免跨库串数据

### 5. 多轮对话

- 通过 `conversation_id` 关联会话上下文
- 默认保留最近 10 轮对话历史
- 支持追问和引导式交互

### 6. 文档切片策略

| 策略 | 适用场景 | 特点 |
|------|----------|------|
| `recursive` | 通用文档、政策文件 | 递归字符切片，按分隔符层级切分 |
| `literature` | 专业书籍、手册、文献 | 更小切片（≤400字），保留语义完整性 |
| `markdown` | 结构化 Markdown 文档 | 按标题层级切片，保留标题元数据 |

### 7. 工具调用

当前已注册工具：

| 工具名 | 说明 | 参数 |
|--------|------|------|
| `fertilizer_calculator` | 肥料用量计算 | `area`（面积/亩）、`crop_type`（作物类型）、`fertilizer_type`（肥料类型） |

新增工具只需继承 `BaseTool` 并调用 `ToolRegistry.register()`：

```python
from app.tools.base import BaseTool, ToolRegistry

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "工具描述"

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {...}}

    async def execute(self, **kwargs) -> dict:
        return {"success": True, "result": ...}

ToolRegistry.register(MyTool())
```

### 8. 快捷推荐卡片

预设 6 张卡片，覆盖核心使用场景：

- 病虫害查询 — 农户田间快速查询
- 价格行情 — 客商查看价格走势
- 施肥计算 — 计算肥料用量
- 政策资讯 — 查询补贴政策
- 天气影响 — 了解天气对作物的影响
- 品种指南 — 了解不同品种特点

### 9. 流式输出

对话接口支持 SSE 流式响应，设置 `stream: true` 即可。响应片段类型：

| type | 说明 |
|------|------|
| `source` | 参考资料来源 |
| `content` | 回答文本片段 |
| `tool` | 工具调用通知 |
| `done` | 生成完成 |

## 支持的文件类型

| 格式 | 扩展名 | 解析方式 |
|------|--------|----------|
| PDF | `.pdf` | pymupdf |
| Word | `.docx`, `.doc` | python-docx |
| 文本 | `.txt`, `.md` | 直接读取 |
| Excel | `.xlsx`, `.xls` | openpyxl |
| CSV | `.csv` | 直接读取 |
| JSON | `.json` | json.loads |

## 配置说明

所有配置通过环境变量或 `.env` 文件管理，详见 `.env.example`：

### MySQL 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MYSQL_HOST` | `localhost` | MySQL 地址 |
| `MYSQL_PORT` | `3306` | MySQL 端口 |
| `MYSQL_USER` | `root` | MySQL 用户名 |
| `MYSQL_PASSWORD` | - | MySQL 密码 |
| `MYSQL_DATABASE` | `orange_rag` | 数据库名 |
| `MYSQL_POOL_SIZE` | `10` | 连接池大小 |
| `MYSQL_MAX_OVERFLOW` | `20` | 连接池最大溢出 |

### 认证与安全配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `JWT_SECRET_KEY` | `change-me-to-a-random-secret-key` | JWT 签名密钥（生产环境务必修改） |
| `JWT_ALGORITHM` | `HS256` | JWT 加密算法 |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | Token 有效期（分钟），默认 24 小时 |
| `CAPTCHA_EXPIRE_SECONDS` | `300` | 验证码有效期（秒），默认 5 分钟 |
| `CAPTCHA_LENGTH` | `4` | 验证码字符数 |

### 其他配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_PROVIDER` | `openai` | LLM 供应商（openai/zhipu/deepseek） |
| `LLM_API_KEY` | - | LLM API 密钥 |
| `LLM_BASE_URL` | OpenAI 默认 | LLM API 地址 |
| `LLM_MODEL_NAME` | `` | 模型名称 |
| `EMBEDDING_PROVIDER` | `openai` | Embedding 供应商 |
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | Embedding 模型 |
| `EMBEDDING_DIMENSION` | `1536` | 向量维度 |
| `QDRANT_HOST` | `localhost` | Qdrant 地址 |
| `QDRANT_PORT` | `6333` | Qdrant 端口 |
| `QDRANT_COLLECTION_NAME` | `orange_knowledge` | 集合名称 |
| `CHUNK_SIZE` | `500` | 切片最大字符数 |
| `CHUNK_OVERLAP` | `50` | 切片重叠字符数 |
| `RAG_TOP_K` | `5` | 检索返回数量 |
| `RAG_SCORE_THRESHOLD` | `0.7` | 相似度阈值 |
| `CHAT_HISTORY_MAX_TURNS` | `10` | 多轮对话保留轮数 |

## 部署

```bash
# 生产环境启动
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

部署到 18 号服务器时，需配置：
1. MySQL 连接信息
2. Qdrant 服务地址
3. LLM API 地址和密钥
4. **修改 `JWT_SECRET_KEY` 为随机强密钥**
5. 申请子域名并配置反向代理
6. CORS `allow_origins` 限制为前端域名
