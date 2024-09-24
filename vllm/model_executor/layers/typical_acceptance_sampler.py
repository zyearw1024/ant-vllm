import torch
import torch.jit

from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeDeterministicBaseSampler)


class TypicalAcceptanceSampler(SpecDecodeDeterministicBaseSampler):
    """Apply typical acceptance sampling as described in section 3.3.1 in 
        "MEDUSA: Simple LLM Inference Acceleration Framework with 
        Multiple Decoding Heads"
        https://arxiv.org/pdf/2401.10774
    """

    def __init__(
        self,
        posterior_threshold: float,
        posterior_alpha: float,
        strict_mode: bool = False,
    ):
        """Create a Typical Acceptance Sampler.

        Args:
            strict_mode: Whether or not to perform shape/device/dtype checks
            during sampling. This catches correctness issues but adds
            nontrivial latency.
            posterior_threshold : A threshold value that sets a lower bound 
            on the posterior probability of a token in target model for it
            to be accepted.
            posterior_alpha : A scaling factor for the entropy-based
            threshold in typical acceptance sampling.
        """
        self._posterior_threshold = posterior_threshold
        self._posterior_alpha = posterior_alpha
        super().__init__(strict_mode=strict_mode)

    def forward(
        self,
        target_with_bonus_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sample token ids using typical acceptance sampling. This accepts 
        or rejects tokens proposed by the draft model using the probability
        of each token according to the draft and target models.

        In the worst case where all draft tokens are rejected, it is guaranteed
        one token will be emitted.

        In the case where all draft tokens are accepted, the bonus token will be
        accepted.

        Args:
            target_probs: The probability distribution over token ids given
                context according to the target model.
            shape = [batch_size, num_speculative_tokens, vocab_size]

            bonus_token_ids: The "bonus" token ids that are accepted iff all
                speculative tokens in a sequence are accepted.
            shape = [batch_size, num_bonus_tokens]

            draft_probs: This parameter is unused by the acceptance sampler.

            draft_token_ids: The token ids that were sampled from the draft
                probabilities.
            shape = [batch_size, num_speculative_tokens]

        Returns:
            output_token_ids: The token ids sampled via rejection sampling,
                or -1 if unable to sample a token because the previous token
                was rejected.
            shape = [batch_size, num_speculative_tokens + num_bonus_tokens]
        """
        # Only perform shape/dtype/device checking in strict mode, as it adds
        # overhead.
        if self._strict_mode:
            self._raise_if_incorrect_input(target_with_bonus_probs,
                                           draft_token_ids, bonus_token_ids)
        target_probs = target_with_bonus_probs[:, :-1]
        accepted = self._evaluate_accepted_tokens(target_probs,
                                                  draft_token_ids)
        recovered_token_ids = self._get_recovered_token_ids(target_probs)
        output_token_ids = self._create_output(accepted, recovered_token_ids,
                                               draft_token_ids,
                                               bonus_token_ids)
        return output_token_ids

    def _evaluate_accepted_tokens(self, target_probs, draft_token_ids):
        r"""
        Evaluates and returns a mask of accepted tokens based on the
        posterior probabilities.

        Parameters:
        ----------
        target_probs : torch.Tensor
            A tensor of shape (batch_size, k, vocab_size) representing 
            the probabilities of each token in the vocabulary for each
            position in the proposed sequence. This is the distribution
            generated by the target model.
        draft_token_ids : torch.Tensor
            A tensor of shape (batch_size, k) representing the proposed
            token ids.

        A draft token_id x_{n+k} is accepted if it satisfies the
        following condition
    
        .. math::
            p_{\text{original}}(x_{n+k} | x_1, x_2, \dots, x_{n+k-1}) > 
            \min \left( \epsilon, \delta * \exp \left(
                -H(p_{\text{original}}(
                    \cdot | x_1, x_2, \ldots, x_{n+k-1})) \right) \right)
        
        where :math:`p_{\text{original}}` corresponds to target_probs 
        and :math:`\epsilon` and :math:`\delta` correspond to hyperparameters
        specified using self._posterior_threshold and self._posterior_alpha

        This method computes the posterior probabilities for the given
        draft token ids based on the provided target probabilities. It
        calculates the entropy of the posterior distribution and determines
        a dynamic threshold for each token position using the provided
        posterior_threshold and posterior_alpha values. The method then
        returns a boolean mask indicating which tokens can be accepted.

        Returns:
        -------
        torch.Tensor
            A boolean tensor of shape (batch_size, k) where each element
            indicates whether the corresponding draft token has been accepted
            or rejected. True indicates acceptance and false indicates
            rejection.
            
        """
        device = target_probs.device
        candidates_prob = torch.gather(
            target_probs, dim=-1,
            index=draft_token_ids.unsqueeze(-1)).squeeze(-1)
        # A small constant added to prevent computing the logarithm of zero,
        # which can lead to undefined values.
        epsilon = 1e-5
        posterior_entropy = -torch.sum(
            target_probs * torch.log(target_probs + epsilon), dim=-1)
        threshold = torch.minimum(
            torch.ones_like(posterior_entropy, device=device) *
            self._posterior_threshold,
            torch.exp(-posterior_entropy) * self._posterior_alpha,
        )
        accepted_mask = candidates_prob > threshold
        return accepted_mask

    def _get_recovered_token_ids(self, target_probs):
        """
        The recovered token ids will fill the first unmatched token
        by the target token.

        Parameters
        ----------
        target_probs : torch.Tensor
            A tensor of shape (batch_size, k, vocab_size) containing 
            the target probability distribution

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, k) with the recovered token
            ids which are selected from target probs.
        """
        max_indices = torch.argmax(target_probs, dim=-1)

        return max_indices
