"""
qi_qiskit_sensing.py

Discrete-variable (qubit-based) toy model of quantum illumination,
implemented in Qiskit, meant as a drop-in sensing module for the
PSO/QPSO cognitive-radio spectrum allocation project.

WHY QUBITS AND NOT THE OPTICAL QI MODEL
----------------------------------------
Published quantum illumination (Lloyd 2008; Tan et al. 2008; the
Shapiro/Pirandola Gaussian-state line of work) is a continuous-variable
protocol built on two-mode squeezed vacuum states of light. Qiskit is a
gate-based, discrete-variable (qubit) framework and does not natively
model continuous-variable bosonic modes, so it cannot literally
reproduce that protocol.

What it CAN do, faithfully, is implement the discrete-variable analogue
of quantum illumination that is studied in its own right in the
literature -- entangled qubit/qudit pairs used for the same "is there a
real signal, or just noise" hypothesis test (see e.g. Kang et al.,
"Single-shot detection limits of quantum illumination with multi-qudit
states," arXiv:2501.14178). That is exactly what this module
implements: a Bell pair standing in for the signal/idler pair, a
depolarizing channel standing in for the lossy/noisy round trip, and a
joint Bell-basis measurement standing in for the optimal joint quantum
receiver. State this explicitly in your methodology section -- it is a
legitimate, citable simplification, not the literal optical protocol.

DESIGN
------
H0 (channel free / nothing transmitted): the qubit that would have
carried information is fully depolarized (p = 1) -- by construction,
there is nothing real to detect.

H1 (channel busy / something transmitted): the same qubit is degraded
by a loss parameter p_loss derived from an assumed sensing SNR (lower
SNR -> higher p_loss -> less of the original correlation survives).

QUANTUM strategy: entangle signal + idler (Bell pair), pass the signal
through the noisy channel, undo the Bell encoding, measure both qubits.
Outcome '00' = "still correlated."

CLASSICAL strategy: send one unentangled qubit (no idler at all)
through the identical noisy channel and measure it. Outcome '0' =
"still looks like what was sent."

Both are run many times ("shots"); the resulting outcome frequencies
are converted into Pd/Pfa using the binomial distribution over a
sensing window of n_window repeated shots, the discrete analogue of
integrating over M modes in the continuous-variable theory.
"""

import numpy as np
from scipy.stats import binom
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

_BACKEND = AerSimulator()


def snr_db_to_ploss(snr_db, floor=0.02, ceiling=0.98):
    """Map an assumed sensing SNR (dB) to a depolarizing 'loss'
    probability for the H1 (signal-present) case. This is a modeling
    choice, not a physical derivation: higher SNR -> lower loss -> more
    correlation survives -> better detection. floor/ceiling just keep
    p_loss numerically sane."""
    gamma = 10 ** (snr_db / 10.0)
    p_loss = 1.0 / (1.0 + gamma)
    return float(np.clip(p_loss, floor, ceiling))


def _quantum_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)        # idler
    qc.cx(0, 1)    # entangle idler (0) with signal (1)
    qc.id(1)       # <- noise model attaches the "channel" here
    qc.cx(0, 1)    # receiver: undo the entangling gate
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    return qc


def _classical_circuit():
    qc = QuantumCircuit(1, 1)
    qc.h(0)        # single unentangled probe qubit
    qc.id(0)       # <- noise model attaches the "channel" here
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _noise_model(p_dep, qubit):
    nm = NoiseModel()
    nm.add_quantum_error(depolarizing_error(p_dep, 1), "id", [qubit])
    return nm


def _run_counts(qc, p_dep, qubit, shots):
    nm = _noise_model(p_dep, qubit)
    qc_t = transpile(qc, basis_gates=["h", "cx", "id"], optimization_level=0)
    job = _BACKEND.run(qc_t, shots=shots, noise_model=nm)
    return job.result().get_counts()


def hit_probabilities(strategy, p_loss, shots):
    """Empirically estimate (p0, p1): probability of the 'still
    correlated' outcome under H0 (nothing transmitted) and H1
    (something transmitted, degraded by p_loss), for a single shot."""
    if strategy == "quantum":
        qc = _quantum_circuit()
        counts_h0 = _run_counts(qc, p_dep=1.0, qubit=1, shots=shots)
        counts_h1 = _run_counts(qc, p_dep=p_loss, qubit=1, shots=shots)
        target = "00"
    elif strategy == "classical":
        qc = _classical_circuit()
        counts_h0 = _run_counts(qc, p_dep=1.0, qubit=0, shots=shots)
        counts_h1 = _run_counts(qc, p_dep=p_loss, qubit=0, shots=shots)
        target = "0"
    else:
        raise ValueError("strategy must be 'quantum' or 'classical'")

    p0 = counts_h0.get(target, 0) / shots
    p1 = counts_h1.get(target, 0) / shots
    return p0, p1


def pd_pfa_from_hit_probs(p0, p1, n_window, target_pfa):
    """Convert single-shot hit probabilities into Pd/Pfa for a sensing
    decision that aggregates n_window repeated shots. Decision rule:
    declare 'busy' if the number of 'hit' outcomes in the window is >=
    threshold tau, where tau is the smallest integer keeping Pfa at or
    below target_pfa."""
    tau = n_window + 1
    for t in range(n_window + 1):
        pfa_t = 1.0 if t == 0 else binom.sf(t - 1, n_window, p0)
        if pfa_t <= target_pfa:
            tau = t
            break
    pfa = 1.0 if tau == 0 else binom.sf(tau - 1, n_window, p0)
    pd = 1.0 if tau == 0 else binom.sf(tau - 1, n_window, p1)
    return pd, pfa


def sensing_pd_pfa(snr_db, strategy="quantum", shots=400, n_window=20,
                    target_pfa=0.05):
    """Drop-in replacement for the analytic sensing_model() used
    earlier in the project: feed it a per-channel SNR in dB, get back
    (Pd, Pfa) -- except these numbers come from an actual simulated
    Qiskit circuit rather than a closed-form formula."""
    p_loss = snr_db_to_ploss(snr_db)
    p0, p1 = hit_probabilities(strategy, p_loss, shots)
    return pd_pfa_from_hit_probs(p0, p1, n_window, target_pfa)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    snr_range = np.arange(-5, 16, 2)
    pd_q, pd_c, pfa_q, pfa_c = [], [], [], []

    for snr in snr_range:
        pdq, pfaq = sensing_pd_pfa(snr, strategy="quantum")
        pdc, pfac = sensing_pd_pfa(snr, strategy="classical")
        pd_q.append(pdq); pfa_q.append(pfaq)
        pd_c.append(pdc); pfa_c.append(pfac)
        print(f"SNR {snr:>3} dB | quantum Pd={pdq:.3f} Pfa={pfaq:.3f} "
              f"| classical Pd={pdc:.3f} Pfa={pfac:.3f}")

    plt.figure(figsize=(6, 5))
    plt.plot(snr_range, pd_q, "o-", label="Quantum (entangled) Pd")
    plt.plot(snr_range, pd_c, "s-", label="Classical (unentangled) Pd")
    plt.xlabel("Sensing SNR (dB)")
    plt.ylabel("Probability of detection")
    plt.title("Qiskit-simulated detection: quantum vs classical strategy")
    plt.legend()
    plt.tight_layout()
    print("\nSaved qiskit_qi_vs_classical.png")
