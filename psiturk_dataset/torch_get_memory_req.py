import torch


def get_tensor_memory_in_mb(shape=(256, 256, 3)):
    data = torch.zeros(shape)
    memory_in_bytes = data.float().element_size() * data.nelement()
    return memory_in_bytes / (1000**3)


def get_total_input_memory(num=700, batch_size=1):
    rgb = get_tensor_memory_in_mb((num, batch_size, 256, 256, 3))
    depth = get_tensor_memory_in_mb((num, batch_size, 256, 256, 1))
    actions = get_tensor_memory_in_mb((num, batch_size))
    print("RGB observations: {}".format(rgb))
    print("Depth observations: {}".format(depth))
    print("Actions: {}".format(actions))
    print("Total memory requirement: {}".format(rgb + depth + actions * 3))


if __name__ == "__main__":
    get_total_input_memory()
