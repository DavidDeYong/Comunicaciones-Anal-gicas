import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from scipy.special import erfc

# ---------- Configuración de la constelación y mapeo Gray ----------
qam_levels = np.array([-3, -1, 1, 3])
iq_grid = (qam_levels[:, None] + 1j * qam_levels).flatten()
iq_grid /= np.sqrt((np.abs(iq_grid) ** 2).mean())

gray_map = {
    (-3, -3): '0000', (-3, -1): '0001', (-3, 1): '0011', (-3, 3): '0010',
    (-1, -3): '0100', (-1, -1): '0101', (-1, 1): '0111', (-1, 3): '0110',
    (1, -3): '1100', (1, -1): '1101', (1, 1): '1111', (1, 3): '1110',
    (3, -3): '1000', (3, -1): '1001', (3, 1): '1011', (3, 3): '1010'
}
reverse_map = {v: k for k, v in gray_map.items()}
bits_lookup = np.array([
    np.array(list(gray_map[(int(np.real(s)*np.sqrt(10)), int(np.imag(s)*np.sqrt(10)))]), dtype=int)
    for s in iq_grid
])

def modulate(bits):
    symbols = []
    for i in range(0, len(bits), 4):
        bit_group = ''.join(bits[i:i+4].astype(str))
        I, Q = reverse_map[bit_group]
        s = complex(I, Q)
        symbols.append(s)
    return np.array(symbols) / np.sqrt(10)

def closest_symbol_indices(received_symbols, constellation):
    distances = np.abs(received_symbols[:, None] - constellation[None, :])**2
    return np.argmin(distances, axis=1)

def demodulate_bayes(received_symbols, sigma2):
    posterior = np.zeros((len(received_symbols), len(iq_grid)))
    for i, s in enumerate(iq_grid):
        posterior[:, i] = -np.abs(received_symbols - s)**2 / sigma2
    decisions = np.argmax(posterior, axis=1)
    return bits_lookup[decisions].reshape(-1)

def simulate(K, snr_db, num_symbols=20000):
    snr_linear = 10**(snr_db / 10)
    noise_power = 1 / snr_linear
    sigma2 = noise_power / 2
    sigma = np.sqrt(sigma2)

    bits_tx = np.random.randint(0, 2, size=num_symbols * 4)
    symbols_tx = modulate(bits_tx)

    A = np.sqrt(2 * K)
    h = np.random.normal(loc=A, scale=sigma, size=num_symbols) + 1j * np.random.normal(0, sigma, size=num_symbols)
    noise = sigma * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
    symbols_rx = h * symbols_tx + noise
    symbols_eq = symbols_rx / h

    bits_rx = demodulate_bayes(symbols_eq, sigma2)
    ber = np.mean(bits_rx != bits_tx[:len(bits_rx)])
    indices_tx = closest_symbol_indices(symbols_tx, iq_grid)

    return symbols_eq, indices_tx, ber

def qam_ber_theoretical(snr_db):
    snr_linear = 10**(snr_db / 10)
    return (3/8) * erfc(np.sqrt((4/5)*snr_linear))

def ber_vs_snr(K, snr_vals, num_symbols=20000, num_trials=10):
    ber_vals = []
    for snr in snr_vals:
        ber_acc = 0
        for _ in range(num_trials):
            _, _, ber = simulate(K, snr, num_symbols=num_symbols)
            ber_acc += ber
        ber_vals.append(ber_acc / num_trials)
    return ber_vals

def interactive_plot():
    fig, (ax, ax_ber) = plt.subplots(2, 1, figsize=(10, 12))
    plt.subplots_adjust(left=0.3, bottom=0.35)

    K_init = 10
    snr_init = 10
    snr_range = np.linspace(0, 20, 41)

    symbols_eq, indices_tx, ber = simulate(K_init, snr_init)
    scatter = ax.scatter(symbols_eq.real, symbols_eq.imag, c=indices_tx, cmap='tab20', alpha=0.6)
    ax.plot(iq_grid.real, iq_grid.imag, 'rx', markersize=10, label='Símbolos ideales')
    ax.set_title(f'K={K_init}, SNR={snr_init} dB, BER={ber:.4e}')
    ax.grid(True)
    ax.axis('equal')
    ax.set_xlabel('In-Phase')
    ax.set_ylabel('Quadrature')

    axK = plt.axes([0.3, 0.25, 0.55, 0.03])
    axSNR = plt.axes([0.3, 0.2, 0.55, 0.03])
    sK = Slider(axK, 'K', 0.0, 10.0, valinit=K_init)
    sSNR = Slider(axSNR, 'SNR (dB)', 0, 20, valinit=snr_init)

    axK_text = plt.axes([0.05, 0.25, 0.2, 0.04])
    axSNR_text = plt.axes([0.05, 0.2, 0.2, 0.04])
    K_textbox = TextBox(axK_text, 'K Input:', initial=str(K_init))
    SNR_textbox = TextBox(axSNR_text, 'SNR Input:', initial=str(snr_init))

    button_ax = plt.axes([0.05, 0.15, 0.2, 0.05])
    run_button = Button(button_ax, 'Ejecutar')

    ber_curve, = ax_ber.semilogy([], [], 'bo-', label='BER simulada')
    ber_theory_curve, = ax_ber.semilogy([], [], 'r--', label='BER teórica AWGN')
    ax_ber.set_xlabel('SNR (dB)')
    ax_ber.set_ylabel('BER')
    ax_ber.set_title('Curva BER vs. SNR')
    ax_ber.grid(True, which='both')
    ax_ber.legend()

    def run_simulation(K, snr_db):
        symbols_eq, indices_tx, ber = simulate(K, snr_db)
        scatter.set_offsets(np.c_[symbols_eq.real, symbols_eq.imag])
        scatter.set_array(indices_tx)
        ax.set_title(f'K={K:.2f}, SNR={snr_db:.0f} dB, BER={ber:.4e}')

        ber_vals = ber_vs_snr(K, snr_range)
        ber_curve.set_data(snr_range, ber_vals)

        ber_theory = qam_ber_theoretical(snr_range)
        ber_theory_curve.set_data(snr_range, ber_theory)

        ax_ber.set_ylim([1e-5, 1])
        ax_ber.set_xlim([snr_range[0], snr_range[-1]])
        fig.canvas.draw_idle()

    def update_slider(val):
        K = sK.val
        snr_db = sSNR.val
        run_simulation(K, snr_db)
        K_textbox.set_val(f"{K:.2f}")
        SNR_textbox.set_val(f"{snr_db:.2f}")

    def on_button_click(event):
        try:
            K_val = float(K_textbox.text)
            SNR_val = float(SNR_textbox.text)
            sK.set_val(K_val)
            sSNR.set_val(SNR_val)
            run_simulation(K_val, SNR_val)
        except ValueError:
            print("Entradas inválidas.")

    sK.on_changed(update_slider)
    sSNR.on_changed(update_slider)
    run_button.on_clicked(on_button_click)

    initial_ber_vals = ber_vs_snr(K_init, snr_range)
    ber_curve.set_data(snr_range, initial_ber_vals)

    ber_theory = qam_ber_theoretical(snr_range)
    ber_theory_curve.set_data(snr_range, ber_theory)

    ax_ber.set_ylim([1e-5, 1])
    ax_ber.set_xlim([snr_range[0], snr_range[-1]])

    plt.show()

# Ejecutar interfaz interactiva
interactive_plot()