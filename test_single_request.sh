#!/bin/bash

# Single Request Test - Policy Document
# This tests a single request to your RAG system

echo "Testing single request to RAG system..."
echo "Document: Policy PDF"
echo "Questions: 10 policy-related questions"
echo "Using localhost server"
echo ""

curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?",
      "Does this policy cover maternity expenses, and what are the conditions?",
      "What is the waiting period for cataract surgery?",
      "Are the medical expenses for an organ donor covered under this policy?",
      "What is the No Claim Discount (NCD) offered in this policy?",
      "Is there a benefit for preventive health check-ups?",
      "How does the policy define a Hospital?",
      "What is the extent of coverage for AYUSH treatments?",
      "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
  }' | jq '.'

echo ""
echo "Single request test completed"
