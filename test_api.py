# test_api.py
try:
    import requests
    import json
except ImportError as e:
    print(f"Erro: {e}")
    print("Instale o pacote requests com: pip install requests")
    exit()

API_URL = "http://localhost:8000"

def test_api():
    # Dados de exemplo (valores do seu dataset)
    sample_data = {
        "features": [
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
            0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
            0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
            0.2654, 0.4601, 0.1189
        ]
    }

    print("Testando API de classificação de câncer...")
    
    try:
        # Teste de saúde da API
        health_check = requests.get(f"{API_URL}/")
        print(f"\nStatus da API: {health_check.status_code}")
        print(f"Resposta: {health_check.json()}\n")

        # Teste de predição
        print("Enviando dados para classificação...")
        response = requests.post(
            f"{API_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=sample_data
        )

        print("\nResultado da predição:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.ConnectionError:
        print("\nErro: Não foi possível conectar à API")
        print("Verifique se a API está rodando (execute python api.py primeiro)")
    except Exception as e:
        print(f"\nErro durante o teste: {str(e)}")

if __name__ == "__main__":
    test_api()