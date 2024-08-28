## multimodel-chatbot
multimodel-chatbotはGPT-4o、Claude、Geminiとのチャット機能と、llamaindexによるPDFのRAG機能を搭載したチャットボットです。
このツールを使用すると、ユーザーは自然言語を使用して PDF ファイルから情報を照会し、関連する回答や要約を取得できます。

### 使用しているライブラリやフレームワーク
- chainlit
  - https://github.com/Chainlit/chainlit
- llamaindex
  - https://www.llamaindex.ai/
 
### セットアップ
1. OpenAI API、Anthropic API、Gemini APIを入手する
2. githubリポジトリをクローン
```bash
$ https://github.com/masui-dev/multimodel-chatbot.git
```
```bash
$ cd multimodel-chatbot
```

3. Dockerfileの3～5行目に各APIキーを入力する
4. docker build & run
```bash
$ docker build ./ -t multi-chainlit
```
```bash
$  docker run -p 8000:8000  multi-chainlit
```

