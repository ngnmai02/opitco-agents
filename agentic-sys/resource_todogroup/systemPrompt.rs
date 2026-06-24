const response = await azureOpenAiClient.chat.completions.create({ 
     model: AI_MODEL, 
     temperature: 0, 
     max_tokens: 400, 
     messages: [ 
       { 
         role: "system", 
         content: AI_EVALUATION_SYSTEM_PROMPT, 
       }, 
       { 
         role: "user", 
         content: JSON.stringify(evaluationPayload), 
       }, 
     ], 
     response_format: { 
       type: "json_schema", 
       json_schema: evaluationResponseJsonSchema, 
     }, 
   }); 