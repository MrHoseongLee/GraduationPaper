def fade(step, total_steps):
    return 2 * step / total_steps if step <= total_steps // 2 else 2 - 2 * step / total_steps
