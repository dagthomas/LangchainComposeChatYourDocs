import { json } from '@sveltejs/kit';
const baseFastapiUrl = import.meta.env.VITE_BASE_FASTAPI_URL
const bearerToken = import.meta.env.VITE_BEARER_TOKEN
import type { RequestHandler } from '@sveltejs/kit';

/** @type {import('./$types').RequestHandler} */
export const POST: RequestHandler = async ({ request }) => {
	const data = await request.json();

	try {
		const response = await fetch(`${baseFastapiUrl}/${data.index}`, {
			method: 'POST',
			headers: {
					'Content-Type': 'application/json',
				Authorization: `Bearer ${bearerToken}`,	
			
			},
			body: JSON.stringify(data),
		});
		if (!response.ok) {
			console.log(response);
		}

		const returnData: any = await response.json();
		return json(returnData);
	} catch (error) {
		return json({ error: 'error' }); //console.error(error)
	}
};
