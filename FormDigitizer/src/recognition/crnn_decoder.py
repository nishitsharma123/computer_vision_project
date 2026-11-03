
import torch


def greedy_decode(logits, vocabulary, input_lengths=None):
    """
    Greedy CTC decoder with optional per-image sequence lengths.

    Args:
        logits:
            Tensor of shape [T, B, C].

        vocabulary:
            Dictionary containing idx_to_char and blank_index.

        input_lengths:
            Actual CTC output length for each image.
            Tensor or list of integers.
            If None, the complete sequence is decoded.

    Returns:
        List of decoded text strings.
    """

    if logits.ndim != 3:
        raise ValueError(
            "Expected logits with shape [T, B, C]"
        )

    idx_to_char = vocabulary["idx_to_char"]
    blank_index = vocabulary["blank_index"]

    time_steps, batch_size, _ = logits.shape

    # Get the most likely class at every timestep.
    predictions = logits.argmax(dim=2)

    # Convert lengths to a Python list.
    if input_lengths is None:

        sequence_lengths = [time_steps] * batch_size

    elif isinstance(input_lengths, torch.Tensor):

        sequence_lengths = (
            input_lengths.detach()
            .cpu()
            .tolist()
        )

    else:

        sequence_lengths = list(input_lengths)

    if len(sequence_lengths) != batch_size:
        raise ValueError(
            "Number of input lengths must equal batch size"
        )

    decoded_texts = []

    for batch_index in range(batch_size):

        actual_length = int(sequence_lengths[batch_index])

        if not 0 <= actual_length <= time_steps:
            raise ValueError(
                f"Invalid sequence length: {actual_length}"
            )

        # IMPORTANT:
        # Decode only the valid timesteps for this image.
        sequence = predictions[
            :actual_length,
            batch_index
        ].tolist()

        output = []

        previous = None

        for token in sequence:

            # CTC: remove blanks and consecutive repetitions.
            if token != blank_index and token != previous:

                character = idx_to_char.get(str(token))

                if character is None:
                    raise KeyError(
                        f"Unknown character index: {token}"
                    )

                output.append(character)

            previous = token

        decoded_texts.append("".join(output))

    return decoded_texts
