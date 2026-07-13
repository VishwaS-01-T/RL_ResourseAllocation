QI-Based Spectrum Sensing Work Summary
What I Built
Implemented a Quantum Illumination (QI)-inspired sensing layer that sits before the PSO/QPSO allocation stage — its job is to decide which of the channels are actually free before they're handed to the allocator, instead of assuming perfect channel-state knowledge.

Built two competing sensing "strategies" in Qiskit: a quantum detector (2-qubit entangled circuit — Hadamard + CNOT idler/signal pair, CNOT+H disentangle before measurement) and a classical baseline (single-qubit H–measure). Both are run through AerSimulator with a depolarizing noise channel standing in for target-return loss.
Signal loss is modeled directly from SNR via snr_db_to_ploss, mapping SNR(dB) → depolarizing probability, clipped to [0.02, 0.98] so neither extreme collapses the model.
Detection is framed as a proper Neyman–Pearson test: for each SNR, I estimate hit probabilities under H0 (no target) and H1 (target present) via repeated circuit shots, then use binom.sf to find the smallest threshold τ on a 30-sample sensing window that keeps false-alarm rate ≤ target Pfa (0.01). Pd and Pfa are computed from that threshold.
Downstream metrics (spectral efficiency, BER, throughput) are derived from Pd/Pfa so sensing quality flows through to a link-layer/allocation-layer payoff, parameterized by a channel bandwidth split at 10,000 channels over 2 GHz and a configurable user-to-channel traffic ratio — this is the hook where PSO/QPSO's channel allocation plugs in downstream.

What I Found — Three Sweeps
Graph 1 — ROC curve at low SNR (-5 dB): Swept target Pfa from 0.001 to 1 (log scale) and recomputed Pd for both strategies at fixed noise. Quantum illumination's ROC curve sits well above the classical baseline across the whole Pfa range at this low-SNR point — the entangled scheme keeps meaningfully higher Pd for the same false-alarm budget, exactly where classical energy detection struggles.
Graph 2 — Pd and BER vs. SNR (-15 dB to 10 dB): This is where the quantum advantage is clearest and most SNR-dependent, not uniform:

At -15 dB both strategies are near-blind (Pd ≈ 0.01–0.02, essentially noise floor).
-5 dB is the interesting regime: quantum Pd ≈ 0.25 vs classical Pd ≈ 0.03 — roughly an 8-9x detection advantage.
At 0 dB the gap is still large: quantum ≈ 0.98 vs classical ≈ 0.48.
By 5 dB both saturate near Pd ≈ 1.0, and by 10 dB they're indistinguishable.
BER follows the same pattern — quantum's BER drops off faster as SNR rises and is roughly an order of magnitude better than classical in the -5 to 0 dB transition band.
Takeaway: the QI advantage isn't constant — it's concentrated in a "low-SNR window" (roughly -10 dB to 0 dB) where classical sensing is unreliable but the quantum scheme still discriminates well. Above ~5 dB, both schemes are already good enough that the advantage disappears — worth flagging since it means the sensing-layer benefit is conditional on the noise regime the PU channels are actually operating in.

Graph 3 — Throughput & spectral efficiency vs. traffic load ratio: Swept the user-to-channel ratio (0.2 to 3.0) at two SNR points (noisy: -5 dB, clean: 10 dB).

At the clean SNR point, quantum and classical throughput/SE curves are essentially on top of each other (both near-perfect Pd there) — no meaningful gap.
At the noisy point, quantum sustains higher throughput across the whole ratio range (e.g. at ratio=0.2, quantum ≈ 0.004 Gbps vs classical ≈ 0.001 Gbps per user — roughly 4x), and both degrade similarly as the ratio grows past 1.0, since bandwidth-per-user shrinks regardless of sensing quality.
This confirms the sensing-layer gain propagates into the allocation-layer payoff, but only materially in the noisy regime — at high traffic load (ratio > 2) both schemes converge toward negligible throughput since there just isn't enough bandwidth per user left, independent of sensing accuracy.

Current State / Next Step
The sensing layer (QI vs classical) is fully characterized across SNR, Pfa target, and traffic load, with a clear result: QI gives a real, order-of-magnitude detection/BER advantage specifically in the -10 dB to 0 dB SNR band, collapsing to parity outside it. The natural next step is to feed the QI-derived Pd/Pfa (instead of assumed perfect sensing) as the channel-occupancy input into Devansh's PSO/QPSO allocator, and re-run the head-to-head allocation comparison under both "QI-sensed" and "classically-sensed" channel maps — that's where we'd see whether better sensing upstream actually changes which allocator (PSO vs QPSO) wins downstream, or just shifts both up together.