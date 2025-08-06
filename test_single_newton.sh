#!/bin/bash

# Alternative Single Request Test - Newton's Principia
# This tests with the physics document

echo "Testing single request to RAG system..."
echo "Document: Newton's Principia PDF"
echo "Questions: 12 physics-related questions"
echo ""

curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
    "questions": [
      "What experiments or observations does Newton describe to support his laws of motion?",
      "How does Newton characterize the concept of force in relation to motion and acceleration?",
      "What role does Newton attribute to centripetal force in maintaining circular motion, and how is it mathematically described?",
      "How does Newton explain the cause of tides according to gravitational interaction?",
      "In what way does Newton differentiate between absolute velocity and relative velocity?",
      "What arguments does Newton provide regarding the elasticity and collision of bodies?",
      "How does Newton use the motion of projectiles to illustrate his three laws of motion?",
      "What is Newtons approach to explaining the stability of the solar system through gravitational forces?",
      "How does Newton argue for the conservation of momentum within interacting bodies?",
      "What is the significance of Newtons law of universal gravitation in explaining the motion of celestial bodies?",
      "How does Newton address the limitations of observational astronomy in his theoretical framework?",
      "What philosophical implications does Newton suggest arise from his mathematical description of nature?"
    ]
  }' | jq '.'

echo ""
echo "Single request test completed"
