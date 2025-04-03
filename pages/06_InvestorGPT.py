import os
import streamlit as st
from typing import Any, Type
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchResults
import requests
from sympy import comp

llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")


class CompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(description="Stock symbol of the company.Examle: AAPL,TSLA")


class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = """
        Use this to get an overview of the financials of the company.
        You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
        Use this to get income statement of a company.
        You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()["annualReports"]


class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformanceTool"
    description = """
        Use this to get the weekly performance of a company stock.
        You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will search for.Example query: Stock Market Symbol for Apple Company"
    )


class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.

    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = (
        StockMarketSymbolSearchToolArgsSchema
    )

    def _run(self, query):
        ddg = DuckDuckGoSearchResults()
        return ddg.run(query)

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="💼",
)

st.markdown(
    """
    # InvestorGPT
            
    Welcome to InvestorGPT.
            
    Write down the name of a company and our Agent will do the research for you.
"""
)

with st.sidebar:
    openai_api_key = st.text_input("Input your OpenAI API Key")

if not openai_api_key:
    st.error("Please input your OpenAI API Key on the sidebar")
else:
    agent = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        tools=[
            StockMarketSymbolSearchTool(),
            CompanyOverviewTool(),
            CompanyIncomeStatementTool(),
            CompanyStockPerformanceTool(),
        ],
        agent_kwargs={
            "system_message": SystemMessage(content="""
                You are a hedge fund manager.
                
                You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
                
                Consider the performance of a stock, the company overview and the income statement.
                
                Be assertive in your judgement and recommend the stock or advise the user against it.
            """)
        }
    )

    company = st.text_input("Write the name of company you are interested on.")

    if company:
        result = agent.invoke(company)
        st.write(result)