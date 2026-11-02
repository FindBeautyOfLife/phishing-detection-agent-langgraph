from __future__ import annotations

from workflow_graph import run_workflow


def _print_help() -> None:
    print("""可用指令：
1. 输入任意 URL 进行检测，例如：http://login-paypal-security-check.xyz
2. 输入多个 URL 进行批量检测
3. 输入“为什么”“依据是什么”“哪里可疑”追问上一次结果
4. 输入“生成报告”“导出表格”“生成风险图表”“生成安全提示图”
5. 输入“你好”“你会干什么”“有什么功能”获取引导
6. 输入 help 或 ? 查看这份帮助
7. 输入 exit 或 quit 退出""")


def _print_file_paths(state: dict) -> None:
    for key in ("report_path", "table_path", "chart_path", "poster_path"):
        if state.get(key):
            print(f"{key}: {state[key]}")


def main() -> None:
    print("钓鱼网站检测与安全报告生成智能体已启动。输入 quit 或 exit 退出。")
    last_detection_results = None

    while True:
        try:
            user_input = input("\n请输入任务：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("已退出。")
            break
        if user_input.lower() in {"help", "?"}:
            _print_help()
            continue

        state = run_workflow(user_input, last_detection_results=last_detection_results)
        if state.get("last_detection_results"):
            last_detection_results = state["last_detection_results"]

        if state.get("error"):
            print(f"[错误] {state['error']}")

        print("\n" + str(state.get("final_answer", "")))
        _print_file_paths(state)


if __name__ == "__main__":
    main()
