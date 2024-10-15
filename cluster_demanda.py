import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar los datos
data_purchases = pd.read_csv('data_purchases_2.csv')
data_sales = pd.read_csv('data_sales.csv', sep=';')
data_items = pd.read_csv('data_items.csv')

# Preprocesar los datos de ventas
data_sales['item_id'] = data_sales['item_id'].astype(int)
data_sales['quantity'] = data_sales['quantity'].astype(int)
data_sales['unit_sale_price (CLP)'] = data_sales['unit_sale_price (CLP)'].astype(float)
data_sales['date'] = pd.to_datetime(data_sales['date'])

# Filtrar ventas de 2023
data_sales_2023 = data_sales[data_sales['date'].dt.year == 2023]

# Crear características de resumen por item para todo el período
sales_summary = data_sales.groupby('item_id').agg(
    total_quantity_sold=('quantity', 'sum'),
    average_price=('unit_sale_price (CLP)', 'mean'),
    sales_count=('item_id', 'count'),
    sales_variance=('quantity', 'var')  # Añadimos la varianza de las ventas para detectar intermitencia
).reset_index()

# Crear características de resumen por item para el año 2023
sales_summary_2023 = data_sales_2023.groupby('item_id').agg(
    total_quantity_sold_2023=('quantity', 'sum'),
    revenue_2023=('total (CLP)', 'sum')  # Sumar el total de las ventas (ingresos) en 2023
).reset_index()

# Unir con los datos de los productos
data = pd.merge(data_items, sales_summary, on='item_id', how='left')
data = pd.merge(data, sales_summary_2023, on='item_id', how='left')

# Reemplazar NaNs por 0 en las características numéricas
features = ['total_quantity_sold', 'average_price', 'sales_count', 'cost (CLP)', 'stock', 'sales_variance', 'total_quantity_sold_2023', 'revenue_2023']
data[features] = data[features].fillna(0)

# 1. Clasificación de demanda intermitente
var_threshold = data['sales_variance'].quantile(0.75)  # Cuartil superior para variación alta
data['demand_type'] = np.where(data['sales_variance'] > var_threshold, 'Intermitente', 'Independiente')

# 2. Clasificación de demanda estacional
data_sales['month'] = data_sales['date'].dt.month
seasonal_sales = data_sales.groupby(['item_id', 'month']).agg(
    monthly_sales=('quantity', 'sum')
).reset_index()

seasonal_summary = seasonal_sales.groupby('item_id').agg(
    max_monthly_sales=('monthly_sales', 'max'),
    total_sales=('monthly_sales', 'sum')
).reset_index()

# Si las ventas en un mes particular superan el 25%, se clasifica como estacional
seasonal_summary['seasonal_ratio'] = seasonal_summary['max_monthly_sales'] / seasonal_summary['total_sales']
seasonal_threshold = 0.25
seasonal_items = seasonal_summary[seasonal_summary['seasonal_ratio'] > seasonal_threshold]['item_id']

# Asignar demanda estacional
data.loc[data['item_id'].isin(seasonal_items), 'demand_type'] = 'Estacional'

# 3. Clasificación de alta y baja rotación
median_sales = data['total_quantity_sold'].median()
data['rotation_type'] = np.where(data['total_quantity_sold'] > median_sales, 'Alta Rotación', 'Baja Rotación')

# 4. Calcular la utilidad para el año 2023 (precio de venta menos costo)
data['profit_margin_2023'] = data['revenue_2023'] - (data['cost (CLP)'] * data['total_quantity_sold_2023'])

# 5. Calcular la utilidad porcentual respecto al total de la empresa en 2023
total_profit_2023 = data['profit_margin_2023'].sum()
data['profit_percent_total'] = (data['profit_margin_2023'] / total_profit_2023) * 100

# 6. Ordenar los productos en orden descendente por la utilidad generada en 2023
data = data.sort_values(by='profit_margin_2023', ascending=False)

# Crear archivos CSV para cada tipo de demanda y rotación
dem_types = ['Intermitente', 'Estacional', 'Independiente']
rot_types = ['Alta Rotación', 'Baja Rotación']

for dem_type in dem_types:
    for rot_type in rot_types:
        demand_data = data[(data['demand_type'] == dem_type) & (data['rotation_type'] == rot_type)]
        output_csv = f"dem_{dem_type.lower()}_{rot_type.lower().replace(' ', '_')}.csv"
        demand_data[['item_id', 'description', 'profit_margin_2023', 'profit_percent_total']].to_csv(output_csv, index=False)

print("Archivos CSV creados con clasificación de demanda, rotación y utilidad porcentual.")
