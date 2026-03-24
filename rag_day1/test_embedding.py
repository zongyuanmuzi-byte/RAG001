from core.embedding import get_embedding

vec = get_embedding("猫吃什么")

print(len(vec))
print(vec[:5])