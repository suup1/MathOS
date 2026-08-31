"""
gui/main_window.py
Main GUI Window — Quant Research Terminal

PySide6 tabbed dashboard with embedded matplotlib charts.
Tabs:
  1. Dashboard (overview + data fetch)
  2. Martingale Analysis
  3. Black-Scholes Pricing
  4. Regime Detection
  5. Sentiment Analysis
"""

import sys
import traceback
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QTextEdit, QSplitter, QGroupBox,
    QScrollArea, QFormLayout, QStatusBar, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QPalette, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0D0D1A;
    color: #E0E0E0;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}
QTabWidget::pane {
    border: 1px solid #2A2A3A;
    background-color: #0D0D1A;
}
QTabBar::tab {
    background-color: #1A1A2E;
    color: #888888;
    padding: 8px 20px;
    border: 1px solid #2A2A3A;
    border-bottom: none;
    min-width: 120px;
}
QTabBar::tab:selected {
    background-color: #0D0D1A;
    color: #00C8FF;
    border-top: 2px solid #00C8FF;
}
QTabBar::tab:hover { color: #E0E0E0; }
QPushButton {
    background-color: #1A1A2E;
    color: #00C8FF;
    border: 1px solid #00C8FF;
    padding: 6px 16px;
    border-radius: 3px;
}
QPushButton:hover {
    background-color: #00C8FF;
    color: #0D0D1A;
    font-weight: bold;
}
QPushButton:disabled { border-color: #444; color: #555; }
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #1A1A2E;
    color: #E0E0E0;
    border: 1px solid #2A2A3A;
    padding: 4px 8px;
    border-radius: 2px;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: #00C8FF; }
QTextEdit {
    background-color: #111122;
    color: #00FF88;
    border: 1px solid #2A2A3A;
    font-family: 'Courier New', monospace;
    font-size: 11px;
}
QGroupBox {
    border: 1px solid #2A2A3A;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
    color: #00C8FF;
    font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
QStatusBar { background-color: #1A1A2E; color: #888; }
QScrollBar:vertical { background: #1A1A2E; width: 8px; }
QScrollBar::handle:vertical { background: #2A2A3A; border-radius: 4px; }
QProgressBar {
    border: 1px solid #2A2A3A;
    border-radius: 3px;
    text-align: center;
    color: #E0E0E0;
}
QProgressBar::chunk { background-color: #00C8FF; }
"""


class WorkerThread(QThread):
    result_ready  = Signal(object)
    error_occurred = Signal(str)
    status_update  = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn     = fn
        self.args   = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}")


class ChartPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.canvas  = None
        self.toolbar = None
        self._placeholder()

    def _placeholder(self):
        fig = Figure(figsize=(10, 5), facecolor="#0D0D1A")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#1A1A2E")
        ax.text(0.5, 0.5, "Run analysis to generate chart",
                ha="center", va="center", color="#444466",
                fontsize=14, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2A2A3A")
        self._set_figure(fig)

    def _set_figure(self, fig):
        for item in [self.toolbar, self.canvas]:
            if item:
                self.layout_.removeWidget(item)
                item.deleteLater()
        self.canvas  = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background: #1A1A2E; color: #888;")
        self.layout_.addWidget(self.toolbar)
        self.layout_.addWidget(self.canvas)

    def update_figure(self, fig):
        self._set_figure(fig)


class DashboardTab(QWidget):
    data_fetched = Signal(object, str) 

    def __init__(self):
        super().__init__()
        self._build_ui()
        self._df = None

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QLabel("◈ MathOS TERMINAL")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Courier New", 18, QFont.Bold))
        header.setStyleSheet("color: #00C8FF; padding: 16px; letter-spacing: 4px;")
        layout.addWidget(header)

        sub = QLabel("Market Data · Quant Models · Regime Detection · Sentiment Analysis")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #555577; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(sub)

        # Fetch controls
        fetch_box = QGroupBox("Market Data Fetch")
        form = QFormLayout(fetch_box)

        self.sym_input      = QLineEdit("AAPL")
        self.lookback_spin  = QSpinBox()
        self.lookback_spin.setRange(30, 1825)
        self.lookback_spin.setValue(365)
        self.lookback_spin.setSuffix(" days")

        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["Daily", "Hourly"])

        form.addRow("Symbol:", self.sym_input)
        form.addRow("Lookback:", self.lookback_spin)
        form.addRow("Timeframe:", self.tf_combo)

        self.fetch_btn = QPushButton("FETCH DATA")
        self.fetch_btn.clicked.connect(self._fetch)
        form.addRow("", self.fetch_btn)

        layout.addWidget(fetch_box)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Stats panel
        self.stats_box = QGroupBox("Data Summary")
        self.stats_layout = QVBoxLayout(self.stats_box)
        self.stats_label  = QLabel("No data loaded.")
        self.stats_label.setStyleSheet("color: #555577; padding: 8px;")
        self.stats_layout.addWidget(self.stats_label)
        layout.addWidget(self.stats_box)

        # Chart
        self.chart = ChartPanel()
        layout.addWidget(self.chart)

    def _fetch(self):
        from data.alpaca_client import AlpacaClient
        from visualization.charts import plot_price_chart
        from alpaca.data.timeframe import TimeFrame

        symbol   = self.sym_input.text().strip().upper()
        lookback = self.lookback_spin.value()
        tf       = TimeFrame.Hour if self.tf_combo.currentText() == "Hourly" else TimeFrame.Day

        self.fetch_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        def _task():
            client = AlpacaClient()
            df     = client.get_bars(symbol, timeframe=tf, lookback_days=lookback)
            return df, symbol

        self._worker = WorkerThread(_task)
        self._worker.result_ready.connect(self._on_data)
        self._worker.error_occurred.connect(lambda e: (
            self.stats_label.setText(f"ERROR:\n{e}"),
            self.stats_label.setStyleSheet("color: #FF4F5A; padding: 8px;"),
            self.fetch_btn.setEnabled(True),
            self.progress.setVisible(False)))
        self._worker.start()

    def _on_data(self, result):
        from visualization.charts import plot_price_chart
        df, symbol = result
        self._df = df

        stats_text = (
            f"Symbol    : {symbol}\n"
            f"Rows      : {len(df)}\n"
            f"From      : {df.index[0].date()}\n"
            f"To        : {df.index[-1].date()}\n"
            f"Last Close: ${df['close'].iloc[-1]:.2f}\n"
            f"52W High  : ${df['close'].max():.2f}\n"
            f"52W Low   : ${df['close'].min():.2f}\n"
            f"Avg Vol   : {df['volume'].mean()/1e6:.2f}M"
        )
        self.stats_label.setText(stats_text)
        self.stats_label.setStyleSheet("color: #00C896; padding: 8px; font-family: monospace;")

        fig = plot_price_chart(df, symbol)
        self.chart.update_figure(fig)

        self.data_fetched.emit(df, symbol)
        self.fetch_btn.setEnabled(True)
        self.progress.setVisible(False)

def _on_error(self, err):
    self.stats_label.setText(f"ERROR:\n{err}")
    self.stats_label.setStyleSheet("color: #FF4F5A; padding: 8px;")
    self.fetch_btn.setEnabled(True)
    self.progress.setVisible(False)
    print(err)


class MartingaleTab(QWidget):
    def __init__(self):
        super().__init__()
        self._df     = None
        self._symbol = "ASSET"
        self._build_ui()

    def load_data(self, df, symbol):
        self._df     = df
        self._symbol = symbol
        self.status.setText(f"Data loaded: {symbol} ({len(df)} rows). Ready.")

    def _build_ui(self):
        layout = QVBoxLayout(self)
        split  = QSplitter(Qt.Horizontal)

        # Left: controls + results
        left = QWidget()
        llay = QVBoxLayout(left)

        ctrl = QGroupBox("Martingale Controls")
        form = QFormLayout(ctrl)

        self.n_paths_spin = QSpinBox()
        self.n_paths_spin.setRange(5, 100)
        self.n_paths_spin.setValue(20)

        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setRange(50, 1000)
        self.n_steps_spin.setValue(252)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.001, 0.5)
        self.sigma_spin.setValue(0.02)
        self.sigma_spin.setSingleStep(0.001)
        self.sigma_spin.setDecimals(3)

        form.addRow("Sim Paths:", self.n_paths_spin)
        form.addRow("Sim Steps:", self.n_steps_spin)
        form.addRow("Sigma:", self.sigma_spin)

        self.run_btn  = QPushButton("▶  RUN MARTINGALE TEST")
        self.run_btn.clicked.connect(self._run)
        form.addRow("", self.run_btn)
        llay.addWidget(ctrl)

        res = QGroupBox("Test Results")
        rlay = QVBoxLayout(res)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(260)
        rlay.addWidget(self.result_text)
        llay.addWidget(res)

        self.status = QLabel("Load data from Dashboard first.")
        self.status.setStyleSheet("color: #555577; padding: 4px;")
        llay.addWidget(self.status)
        llay.addStretch()

        # Right: charts
        right = QWidget()
        rlay2 = QVBoxLayout(right)

        self.tab_sub = QTabWidget()
        self.chart_sim  = ChartPanel()
        self.chart_dist = ChartPanel()
        self.tab_sub.addTab(self.chart_sim,  "Simulation Paths")
        self.tab_sub.addTab(self.chart_dist, "Return Distribution")
        rlay2.addWidget(self.tab_sub)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([300, 700])
        layout.addWidget(split)

    def _run(self):
        from models.martingale import run_martingale_report, simulate_martingale
        from visualization.charts import plot_martingale_paths, plot_martingale_returns

        if self._df is None:
            self.result_text.setText("No data loaded. Fetch data from Dashboard first.")
            return

        self.run_btn.setEnabled(False)
        prices  = self._df["close"]

        def _task():
            report = run_martingale_report(prices, self._symbol)
            sim    = simulate_martingale(
                S0     = float(prices.iloc[0]),
                sigma  = self.sigma_spin.value(),
                n_steps= self.n_steps_spin.value(),
                n_paths= self.n_paths_spin.value()
            )
            fig_paths = plot_martingale_paths(sim, prices, self._symbol)
            fig_dist  = plot_martingale_returns(
                np.log(prices / prices.shift(1)).dropna(), self._symbol
            )
            return report, fig_paths, fig_dist

        self._worker = WorkerThread(_task)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(lambda e: self.result_text.setText(e))
        self._worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self._worker.start()

    def _on_result(self, data):
        report, fig_paths, fig_dist = data

        text_lines = [
            f"MARTINGALE ANALYSIS — {report['symbol']}",
            f"Observations: {report['n_observations']}",
            "",
            "[ Basic Test ]",
            f"  Mean Return : {report['basic_test']['mean_return']}",
            f"  T-Statistic : {report['basic_test']['t_stat']}",
            f"  P-Value     : {report['basic_test']['p_value']}",
            f"  Verdict     : {report['basic_test']['verdict']}",
            "",
            "[ ADF Test (Unit Root) ]",
            f"  ADF Stat    : {report['adf_test']['adf_stat']}",
            f"  P-Value     : {report['adf_test']['p_value']}",
            f"  Verdict     : {report['adf_test']['verdict']}",
            "",
            "[ Variance Ratio Test (q=4) ]",
            f"  VR Ratio    : {report['variance_ratio']['VR']}",
            f"  Z-Statistic : {report['variance_ratio']['z_stat']}",
            f"  P-Value     : {report['variance_ratio']['p_value']}",
            f"  Verdict     : {report['variance_ratio']['verdict']}",
        ]
        self.result_text.setText("\n".join(text_lines))
        self.chart_sim.update_figure(fig_paths)
        self.chart_dist.update_figure(fig_dist)
        self.status.setText("Analysis complete.")


# ──────────────────────────────────────────────
#  TAB 3: BLACK-SCHOLES
# ──────────────────────────────────────────────
class BlackScholesTab(QWidget):
    def __init__(self):
        super().__init__()
        self._df = None
        self._build_ui()

    def load_data(self, df, symbol):
        self._df = df
        # Auto-fill spot price
        self.S_spin.setValue(float(df["close"].iloc[-1]))

    def _build_ui(self):
        layout = QVBoxLayout(self)
        split  = QSplitter(Qt.Horizontal)

        # Controls
        left = QWidget()
        llay = QVBoxLayout(left)

        ctrl = QGroupBox("Black-Scholes Inputs")
        form = QFormLayout(ctrl)

        def dbl(val, lo=0.01, hi=100000, step=0.5, dec=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(dec)
            return s

        self.S_spin     = dbl(150.0, dec=2)
        self.K_spin     = dbl(155.0, dec=2)
        self.T_spin     = dbl(0.25,  lo=0.001, hi=10, step=0.01, dec=3)
        self.r_spin     = dbl(0.05,  lo=0.0,   hi=1,  step=0.005, dec=4)
        self.sigma_spin = dbl(0.20,  lo=0.001, hi=5,  step=0.01,  dec=4)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["call", "put"])

        form.addRow("Spot (S):", self.S_spin)
        form.addRow("Strike (K):", self.K_spin)
        form.addRow("Expiry T (years):", self.T_spin)
        form.addRow("Risk-Free Rate r:", self.r_spin)
        form.addRow("Volatility σ:", self.sigma_spin)
        form.addRow("Option Type:", self.type_combo)

        self.run_btn = QPushButton("▶  PRICE OPTION")
        self.run_btn.clicked.connect(self._run)
        form.addRow("", self.run_btn)
        llay.addWidget(ctrl)

        res = QGroupBox("Pricing Results")
        rlay = QVBoxLayout(res)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        rlay.addWidget(self.result_text)
        llay.addWidget(res)
        llay.addStretch()

        # Charts
        right = QWidget()
        rlay2 = QVBoxLayout(right)
        self.tab_sub = QTabWidget()
        self.chart_surf   = ChartPanel()
        self.chart_greeks = ChartPanel()
        self.tab_sub.addTab(self.chart_surf,   "Price Surface")
        self.tab_sub.addTab(self.chart_greeks, "Greeks")
        rlay2.addWidget(self.tab_sub)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([300, 700])
        layout.addWidget(split)

    def _run(self):
        from models.black_scholes import run_bs_report, generate_vol_surface
        from visualization.charts import plot_bs_surface, plot_greeks

        S     = self.S_spin.value()
        K     = self.K_spin.value()
        T     = self.T_spin.value()
        r     = self.r_spin.value()
        sigma = self.sigma_spin.value()
        otype = self.type_combo.currentText()

        self.run_btn.setEnabled(False)

        def _task():
            report = run_bs_report(S, K, T, r, sigma)
            surf   = generate_vol_surface(S, r, sigma, otype)
            fig_s  = plot_bs_surface(surf, otype)
            fig_g  = plot_greeks(report[otype]["greeks"], otype)
            return report, fig_s, fig_g

        self._worker = WorkerThread(_task)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(lambda e: self.result_text.setText(e))
        self._worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self._worker.start()

    def _on_result(self, data):
        report, fig_s, fig_g = data
        otype = self.type_combo.currentText()
        side  = report[otype]
        inp   = report["inputs"]

        lines = [
            f"BLACK-SCHOLES PRICING",
            f"{'─'*40}",
            f"Spot       : ${inp['S']:.2f}",
            f"Strike     : ${inp['K']:.2f}",
            f"Expiry     : {inp['T']:.4f} yr",
            f"Rate       : {inp['r']*100:.2f}%",
            f"Volatility : {inp['sigma']*100:.2f}%",
            f"Moneyness  : {inp['moneyness']}",
            f"{'─'*40}",
            f"CALL Price : ${report['call']['price']:.4f}",
            f"PUT  Price : ${report['put']['price']:.4f}",
            f"{'─'*40}",
            f"d1 : {side['d1']:.4f}    d2 : {side['d2']:.4f}",
            f"{'─'*40}",
            f"GREEKS ({otype.upper()})",
            f"  Delta : {side['greeks']['delta']:.4f}",
            f"  Gamma : {side['greeks']['gamma']:.6f}",
            f"  Theta : {side['greeks']['theta']:.4f} (per day)",
            f"  Vega  : {side['greeks']['vega']:.4f} (per 1% σ)",
            f"  Rho   : {side['greeks']['rho']:.4f} (per 1% r)",
            f"{'─'*40}",
            f"Put-Call Parity Check : {report['put_call_parity_check']:.8f} (≈0 = valid)",
        ]
        self.result_text.setText("\n".join(lines))
        self.chart_surf.update_figure(fig_s)
        self.chart_greeks.update_figure(fig_g)


class RegimeTab(QWidget):
    def __init__(self):
        super().__init__()
        self._df     = None
        self._symbol = "ASSET"
        self._build_ui()

    def load_data(self, df, symbol):
        self._df     = df
        self._symbol = symbol
        self.status.setText(f"Data loaded: {symbol}")

    def _build_ui(self):
        layout = QVBoxLayout(self)
        split  = QSplitter(Qt.Horizontal)

        left = QWidget()
        llay = QVBoxLayout(left)

        ctrl = QGroupBox("HMM Configuration")
        form = QFormLayout(ctrl)

        self.n_states = QSpinBox()
        self.n_states.setRange(2, 4)
        self.n_states.setValue(2)
        self.n_iter   = QSpinBox()
        self.n_iter.setRange(100, 5000)
        self.n_iter.setValue(1000)
        self.n_iter.setSingleStep(100)

        form.addRow("Hidden States:", self.n_states)
        form.addRow("Max Iterations:", self.n_iter)

        self.run_btn = QPushButton("▶  DETECT REGIMES")
        self.run_btn.clicked.connect(self._run)
        form.addRow("", self.run_btn)
        llay.addWidget(ctrl)

        res = QGroupBox("Regime Statistics")
        rlay = QVBoxLayout(res)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        rlay.addWidget(self.result_text)
        llay.addWidget(res)

        self.status = QLabel("Load data from Dashboard first.")
        self.status.setStyleSheet("color: #555577; padding: 4px;")
        llay.addWidget(self.status)
        llay.addStretch()

        right = QWidget()
        rlay2 = QVBoxLayout(right)
        self.chart = ChartPanel()
        rlay2.addWidget(self.chart)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([300, 700])
        layout.addWidget(split)

    def _run(self):
        from models.regime import fit_hmm
        from visualization.charts import plot_regime_chart

        if self._df is None:
            self.result_text.setText("No data loaded.")
            return

        self.run_btn.setEnabled(False)
        prices   = self._df["close"]
        n_states = self.n_states.value()
        n_iter   = self.n_iter.value()
        symbol   = self._symbol

        def _task():
            result = fit_hmm(prices, n_states=n_states, n_iter=n_iter)
            fig    = plot_regime_chart(result, symbol)
            return result, fig

        self._worker = WorkerThread(_task)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(lambda e: self.result_text.setText(e))
        self._worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self._worker.start()

    def _on_result(self, data):
        result, fig = data
        lines = [
            f"REGIME DETECTION — {self._symbol}",
            f"States   : {result['n_states']}",
            f"Log-Lik  : {result['score']}",
            f"Current  : {result['current_state']}",
            "",
            "Current Probabilities:",
        ]
        for state, prob in result["current_probs"].items():
            lines.append(f"  {state:<10} : {prob:.4f}")
        lines += ["", "Regime Statistics:"]
        for regime, s in result["stats"].items():
            lines += [
                f"  [{regime}]",
                f"    Mean Daily Return : {s['mean_daily_return']}%",
                f"    Annualized Vol    : {s['annualized_vol']}%",
                f"    Days in Regime    : {s['n_days']} ({s['pct_of_time']}%)",
            ]
        lines += ["", "Transition Matrix:", result["transition_matrix"].to_string()]
        self.result_text.setText("\n".join(lines))
        self.chart.update_figure(fig)

class SentimentTab(QWidget):
    def __init__(self):
        super().__init__()
        self._df     = None
        self._symbol = "ASSET"
        self._build_ui()

    def load_data(self, df, symbol):
        self._df     = df
        self._symbol = symbol
        self.sym_input.setText(symbol)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        split  = QSplitter(Qt.Horizontal)

        left = QWidget()
        llay = QVBoxLayout(left)

        ctrl = QGroupBox("Sentiment Config")
        form = QFormLayout(ctrl)

        self.sym_input   = QLineEdit("AAPL")
        self.name_input  = QLineEdit("Apple")
        self.days_spin   = QSpinBox()
        self.days_spin.setRange(1, 30)
        self.days_spin.setValue(7)
        self.days_spin.setSuffix(" days")
        self.arts_spin   = QSpinBox()
        self.arts_spin.setRange(5, 100)
        self.arts_spin.setValue(50)

        form.addRow("Symbol:", self.sym_input)
        form.addRow("Company Name:", self.name_input)
        form.addRow("Days Back:", self.days_spin)
        form.addRow("Max Articles:", self.arts_spin)

        self.run_btn = QPushButton("▶  FETCH SENTIMENT")
        self.run_btn.clicked.connect(self._run)
        form.addRow("", self.run_btn)
        llay.addWidget(ctrl)

        self.demo_btn = QPushButton("◈  USE DEMO DATA (no API)")
        self.demo_btn.setStyleSheet("color: #FFD700; border-color: #FFD700;")
        self.demo_btn.clicked.connect(self._demo)
        llay.addWidget(self.demo_btn)

        res = QGroupBox("Sentiment Summary")
        rlay = QVBoxLayout(res)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        rlay.addWidget(self.result_text)
        llay.addWidget(res)
        llay.addStretch()

        right = QWidget()
        rlay2 = QVBoxLayout(right)
        self.chart = ChartPanel()
        rlay2.addWidget(self.chart)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([300, 700])
        layout.addWidget(split)

    def _demo(self):
        from sentiment.sentiment_engine import demo_sentiment
        from visualization.charts import plot_sentiment_overlay

        if self._df is None:
            self.result_text.setText("Load price data from Dashboard first.")
            return

        demo_s = demo_sentiment(n=len(self._df))
        demo_s.index = self._df.index[:len(demo_s)]

        self.result_text.setText(
            "DEMO MODE — Synthetic Sentiment\n"
            "─────────────────────────────\n"
            f"Points  : {len(demo_s)}\n"
            f"Mean    : {demo_s.mean():.4f}\n"
            f"Std     : {demo_s.std():.4f}\n"
            f"Min     : {demo_s.min():.4f}\n"
            f"Max     : {demo_s.max():.4f}\n\n"
            "Note: Use NewsAPI key for real sentiment."
        )
        fig = plot_sentiment_overlay(self._df, demo_s, self._symbol)
        self.chart.update_figure(fig)

    def _run(self):
        from sentiment.sentiment_engine import run_sentiment_pipeline
        from visualization.charts import plot_sentiment_overlay

        if self._df is None:
            self.result_text.setText("Load price data from Dashboard first.")
            return

        symbol = self.sym_input.text().strip().upper()
        name   = self.name_input.text().strip()
        days   = self.days_spin.value()
        arts   = self.arts_spin.value()
        df     = self._df

        self.run_btn.setEnabled(False)

        def _task():
            result = run_sentiment_pipeline(symbol, name, days, arts)
            return result

        def _on_result(result):
            agg  = result["aggregate"]
            sent = result.get("daily_sentiment_smooth", pd.Series(dtype=float))

            lines = [
                f"SENTIMENT ANALYSIS — {symbol}",
                f"Articles    : {agg['n']}",
                f"Compound    : {agg['compound']:.4f}",
                f"Signal      : {agg['label']}",
                f"Bullish     : {agg.get('bullish_pct', 0)}%",
                f"Bearish     : {agg.get('bearish_pct', 0)}%",
                f"Neutral     : {agg.get('neutral_pct', 0)}%",
            ]

            if not result["articles_df"].empty:
                lines += ["", "Recent Headlines:"]
                for _, row in result["articles_df"].head(5).iterrows():
                    lines.append(f"  [{row.get('label','?'):7}] {str(row['title'])[:70]}")

            self.result_text.setText("\n".join(lines))

            if not sent.empty:
                fig = plot_sentiment_overlay(df, sent, symbol)
            else:
                from sentiment.sentiment_engine import demo_sentiment
                demo_s = demo_sentiment(n=len(df))
                demo_s.index = df.index[:len(demo_s)]
                fig = plot_sentiment_overlay(df, demo_s, symbol)
                self.result_text.append("\n[Note: Using demo data — no daily sentiment returned]")

            self.chart.update_figure(fig)
            self.run_btn.setEnabled(True)

        self._worker = WorkerThread(_task)
        self._worker.result_ready.connect(_on_result)
        self._worker.error_occurred.connect(lambda e: (self.result_text.setText(e),
                                                        self.run_btn.setEnabled(True)))
        self._worker.start()


class QuantApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("◈ MathOS Terminal")
        self.setGeometry(80, 80, 1400, 900)
        self.setStyleSheet(DARK_STYLE)
        self._build_ui()

    def _build_ui(self):
        self.tabs = QTabWidget()

        self.tab_dash  = DashboardTab()
        self.tab_mart  = MartingaleTab()
        self.tab_bs    = BlackScholesTab()
        self.tab_reg   = RegimeTab()
        self.tab_sent  = SentimentTab()

        self.tabs.addTab(self.tab_dash,  "⬡  Dashboard")
        self.tabs.addTab(self.tab_mart,  "⬡  Martingale")
        self.tabs.addTab(self.tab_bs,    "⬡  Black-Scholes")
        self.tabs.addTab(self.tab_reg,   "⬡  Regime")
        self.tabs.addTab(self.tab_sent,  "⬡  Sentiment")

        # Wire data signal
        self.tab_dash.data_fetched.connect(self._on_data_loaded)

        self.setCentralWidget(self.tabs)

        # Status bar
        self.status = QStatusBar()
        self.status.showMessage("MathOS - Ready")
        self.setStatusBar(self.status)

    def _on_data_loaded(self, df, symbol):
        """Distribute fetched data to all tabs."""
        self.tab_mart.load_data(df, symbol)
        self.tab_bs.load_data(df, symbol)
        self.tab_reg.load_data(df, symbol)
        self.tab_sent.load_data(df, symbol)
        self.status.showMessage(
            f"Data loaded: {symbol} — {len(df)} bars — "
            f"{df.index[0].date()} to {df.index[-1].date()}"
        )
