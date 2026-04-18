from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "neo4j://127.0.0.1:7687",
    auth=("neo4j", "mypassword123")
)

with driver.session() as session:
    result = session.run("RETURN 'Connected' AS msg")
    print(result.single()["msg"])