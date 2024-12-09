import os
import streamlit as st
import streamlit_authenticator as stauth


class streamlit_utility(object):
    def __init__(self):
        pass

    def environment_settings(self):
        os.environ["LANGFUSE_PUBLIC_KEY"] = st.secrets.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = st.secrets.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_HOST"] = st.secrets.LANGFUSE_HOST
        os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY

    def login_settings(self):
        _secrets_to_config = {}
        _secrets_to_config["usernames"] = {
            st.secrets["login"]["user"]: {
                "email": st.secrets["login"]["email"],
                "name": st.secrets["login"]["name"],
                "password": st.secrets["login"]["password"],
            }
        }
        return _secrets_to_config

    def login_setup(self, secrets_to_config):
        authenticator = stauth.Authenticate(
            secrets_to_config,
            st.secrets["login"]["name"],
            st.secrets["cookie"]["key"],
            st.secrets["cookie"]["expiry_days"],
        )
        return authenticator

    def initiate_authentication(self):
        _login_config = self.login_settings()
        authenticator = self.login_setup(_login_config)
        return authenticator

    def markdown_style(self) -> str:
        return """<style>
            div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-family: 'Roboto', sans-serif;
            font-size: 18px;
            font-weight: 500;
            color: #091747;
            }
            </style>
        """
