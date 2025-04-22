.PHONY: all clean

all: run

run:
	@echo "Running data fetching..."
	python3 1_create_database.py
	@echo "\nRunning feature calculation..."
	python3 2_calculate_features.py
	@echo "\nRunning EDA plot generation..."
	python3 3_create_plots.py
	@echo "\nRunning Random Forest prediction model..."
	python3 4_prediction_model.py
	@echo "\nRunning price direction prediction..."
	python3 5_predict_price_direction.py
	@echo "\nRunning LSTM prediction model..."
	python3 6_lstm_prediction.py
	@echo "\n✅ Pipeline finished."

clean:
	@echo "Cleaning generated files..."
	rm -f stocks.db
	rm -f *.png
	rm -rf plots/*.png
	@echo "🧹 Clean complete."
