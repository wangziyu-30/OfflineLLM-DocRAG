from rag.chain import rag_chain_with_memory, clear_session_history

if __name__ == "__main__":
    # 测试会话ID
    test_session_id = "test_user_001"

    # 第一轮对话
    print("第一轮问题：千问大模型的上下文窗口是多少？")
    response1 = rag_chain_with_memory.invoke(
        input={"question": "千问大模型的上下文窗口是多少？"},
        config={"configurable": {"session_id": test_session_id}}
    )
    print("回答：", response1)
    print("-" * 50)

    # 第二轮对话：测试上下文记忆，问题中用“它”指代豆包大模型
    print("第二轮问题：它的温度参数怎么设置？")
    response2 = rag_chain_with_memory.invoke(
        input={"question": "它的温度参数怎么设置？"},
        config={"configurable": {"session_id": test_session_id}}
    )
    print("回答：", response2)
    print("-" * 50)
    print("test.txt里写了什么？")
    response3 = rag_chain_with_memory.invoke(
        input={"question": "test.txt文件里写了什么"},
        config={"configurable": {"session_id": test_session_id}}
    )
    print("回答：", response3)
    # 清空会话历史
    clear_session_history(test_session_id)
    print("会话历史已清空")