import { SvelteKitAuth } from '@auth/sveltekit';
import Auth0 from '@auth/core/providers/auth0';
import type { Profile } from '@auth/core/types';
import type { Provider } from '@auth/core/providers';
import { sequence } from '@sveltejs/kit/hooks';
import { redirect, type Handle } from '@sveltejs/kit';

/*
HANDLE REQUESTS
*/
export const base = (async ({ event, resolve }) => {
	const response = await resolve(event, {
		transformPageChunk: ({ html }) => html
	});

	return response;
}) satisfies Handle;

export let auth = SvelteKitAuth({
	providers: [
		Auth0({
			// authorization: {
			// 	params: {
			// 		//   redirect_uri: redirectUrl,
			// 		scope: auth0Scope,
			// 		audience: auth0Audience
			// 	}
			// },
			clientId: '',
			clientSecret: '',
			issuer: ''
		}) as Provider<Profile>
	],
	trustHost: true,
	callbacks: {},
	secret: 'my secret in numbers'
});
//export const handle: Handle = sequence(base)
async function authorization({ event, resolve }) {
	if (event.url.pathname.indexOf('/sql') !== -1) {
		const session = await event.locals.getSession();
		console.log(session);
		if (!session) {
			throw redirect(303, '/');
		}
	}
	const result = await resolve(event, {
		transformPageChunk: ({ html }) => html
	});
	return result;
}
export const handle: Handle = base;
