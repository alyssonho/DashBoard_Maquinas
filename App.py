import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    # Carregar o dataset
    df = pd.read_csv("smart_manufacturing_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filtrar máquinas com falha ou status Normal
    machines_with_issues_or_normal = df[
        df['failure_type'].notna() | (df['failure_type'] == 'Normal')
    ]['machine'].unique()

    # Selecionar as primeiras 5 máquinas como default
    default_machines = machines_with_issues_or_normal[:5] if len(machines_with_issues_or_normal) >= 5 else machines_with_issues_or_normal

    # Sidebar - Filtros
    st.sidebar.header("Filtros")
    selected_machines = st.sidebar.multiselect(
        "Selecione as máquinas:",
        df['machine'].unique(),
        default=default_machines
    )
    date_range = st.sidebar.date_input(
        "Selecione o intervalo de datas:",
        [df['timestamp'].min(), df['timestamp'].max()]
    )
    failure_filter = st.sidebar.multiselect(
        "Tipo de falha:",
        df['failure_type'].dropna().unique(),
        default=df['failure_type'].dropna().unique()
    )

    # Aplicar os filtros
    filtered_df = df[
        (df['machine'].isin(selected_machines)) &
        (df['timestamp'].dt.date >= date_range[0]) &
        (df['timestamp'].dt.date <= date_range[1]) &
        (df['failure_type'].isin(failure_filter))
    ]

    # Abas
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Análises Avançadas", "🔍 Manutenção"])

    with tab1:
        st.header("Dashboard de Indicadores")
        try:
            # Gráfico 1: Temperatura Média por Máquina
            fig1, ax1 = plt.subplots()
            temp_media = filtered_df.groupby('machine')['temperature'].mean()
            ax1.bar(temp_media.index, temp_media.values)
            ax1.set_ylabel("Temperatura Média")
            ax1.set_xlabel("Máquina")
            st.pyplot(fig1)
        except Exception as e:
            st.error(f"Erro ao gerar Gráfico 1: {e}")

        try:
            # Gráfico 2: Variação de Vibração
            fig2, ax2 = plt.subplots()
            ax2.boxplot(
                [filtered_df[filtered_df['machine'] == m]['vibration'] for m in selected_machines],
                labels=selected_machines
            )
            ax2.set_ylabel("Vibração")
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Erro ao gerar Gráfico 2: {e}")

        try:
            # Novo Gráfico 3: Máquinas Precisando de Manutenção
            fig3, ax3 = plt.subplots()
            maintenance_counts = filtered_df.groupby(['machine', 'maintenance_required']).size().unstack(fill_value=0)
            maintenance_counts.plot(kind='bar', stacked=True, ax=ax3)
            ax3.set_ylabel("Quantidade de Registros")
            ax3.set_xlabel("Máquina")
            ax3.set_title("Máquinas Precisando de Manutenção")
            st.pyplot(fig3)
        except Exception as e:
            st.error(f"Erro ao gerar Gráfico 3: {e}")

    with tab2:
        st.header("Análises Avançadas")
        try:
            # Gráfico 4: Dispersão Temperatura vs Vibração
            fig4, ax4 = plt.subplots()
            scatter = ax4.scatter(
                filtered_df['temperature'],
                filtered_df['vibration'],
                c=filtered_df['pressure'],
                cmap='viridis'
            )
            ax4.set_xlabel("Temperatura")
            ax4.set_ylabel("Vibração")
            plt.colorbar(scatter, ax=ax4, label='Pressão')
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Erro ao gerar Gráfico 4: {e}")

        try:
            # Gráfico 5: Histograma da Umidade
            fig5, ax5 = plt.subplots()
            ax5.hist(filtered_df['humidity'], bins=20)
            ax5.set_xlabel("Umidade")
            ax5.set_ylabel("Frequência")
            st.pyplot(fig5)
        except Exception as e:
            st.error(f"Erro ao gerar Gráfico 5: {e}")

        try:
            # Gráfico 6: Predição de Vida Útil
            fig6, ax6 = plt.subplots()
            ax6.boxplot(filtered_df['predicted_remaining_life'])
            ax6.set_ylabel("Vida Útil Prevista")
            st.pyplot(fig6)
        except Exception as e:
            st.error(f"Erro ao gerar Gráfico 6: {e}")

    with tab3:
        st.header("Análises de Manutenção")
        try:
            # Gráfico 7: Falhas por Tipo
            fig7, ax7 = plt.subplots()
            failure_counts = filtered_df['failure_type'].value_counts()
            ax7.pie(failure_counts, labels=failure_counts.index, autopct='%1.1f%%', startangle=90)
            ax7.axis('equal')
            st.pyplot(fig7)
        except Exception as e:
            st.error(f"Erro ao gerar Gráfico 7: {e}")

        try:
            # Gráfico 8: Risco de Parada
            fig8, ax8 = plt.subplots()
            ax8.bar(filtered_df['machine'], filtered_df['downtime_risk'].fillna(0))
            ax8.set_ylabel("Risco de Parada")
            st.pyplot(fig8)
        except Exception as e:
            st.error(f"Erro ao gerar Gráfico 8: {e}")

    # Página adicional para dados
    with st.sidebar:
        st.markdown("---")
        if st.sidebar.button("📄 Ver Dados e Baixar"):
            st.session_state.show_data_page = True

    if 'show_data_page' in st.session_state and st.session_state.show_data_page:
        st.header("📄 Dados")
        st.dataframe(filtered_df)
        try:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Baixar Dados como CSV", csv, "filtered_data.csv", "text/csv")
        except Exception as e:
            st.error(f"Erro ao exportar dados: {e}")

except FileNotFoundError:
    st.error("Arquivo de dados não encontrado. Por favor, coloque o arquivo 'smart_manufacturing_data.csv' na mesma pasta do aplicativo.")
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar o aplicativo: {e}")
