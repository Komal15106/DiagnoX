from flask import Flask, request, jsonify, send_from_directory
import pickle, os, numpy as np
from disease_data import DISEASE_DETAILS

print("Imports successful")
print(f"Disease count: {len(DISEASE_DETAILS)}")
