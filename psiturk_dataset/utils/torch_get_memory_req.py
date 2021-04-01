import torch


def get_tensor_memory_in_gb(shape=(256, 256, 3)):
    data = torch.zeros(shape)
    memory_in_bytes = data.float().element_size() * data.nelement()
    return memory_in_bytes / (1024**3)


def get_total_input_memory(num=1500, batch_size=1):
    rgb = get_tensor_memory_in_gb((num, batch_size, 256, 256, 3))
    depth = get_tensor_memory_in_gb((num, batch_size, 256, 256, 1))
    actions = get_tensor_memory_in_gb((num, batch_size))
    instructions = get_tensor_memory_in_gb((num, batch_size, 256))
    print("RGB observations: {}".format(rgb))
    print("Depth observations: {}".format(depth))
    print("Actions: {}".format(actions))
    print("Instructions: {}".format(instructions))
    print("Total memory requirement: {}".format(rgb + depth + actions * 2 + instructions))


if __name__ == "__main__":
    get_total_input_memory()
